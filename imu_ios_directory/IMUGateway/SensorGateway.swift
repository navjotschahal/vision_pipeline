import ARKit
import CoreMotion
import Combine
import CoreImage
import Foundation
import Network
import UIKit

final class SensorGateway: NSObject, ObservableObject {
    @Published private(set) var isStreaming = false
    @Published private(set) var imuConnectionState = "Not connected"
    @Published private(set) var rgbdConnectionState = "Not connected"
    @Published private(set) var depthCapability = "Checking..."
    @Published private(set) var samplesSent: UInt64 = 0
    @Published private(set) var framesSent: UInt64 = 0
    @Published private(set) var framesDropped: UInt64 = 0
    @Published var host = "192.168.1.100"
    @Published var port = "5001"

    private let motionManager = CMMotionManager()
    private let motionQueue = OperationQueue()
    private let frameQueue = DispatchQueue(label: "com.visionpipeline.rgbd.capture", qos: .userInitiated)
    private let arSession = ARSession()
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])
    private let encoder = JSONEncoder()
    private var imuConnection: NWConnection?
    private var rgbdConnection: NWConnection?
    private var imuSequenceNumber: UInt64 = 0
    private var frameSequenceNumber: UInt64 = 0
    private var previousFrameTimestamp = -Double.infinity
    private var rgbdConnectionReady = false
    private var frameSendPending = false
    private var wasStreamingBeforeBackground = false
    private let frameIntervalSeconds = 1.0 / 10.0
    private let sourceID = "iphone-\(UIDevice.current.identifierForVendor?.uuidString.lowercased() ?? UUID().uuidString.lowercased())"
    private var sessionID = UUID().uuidString.lowercased()
    private lazy var frameDelegate = RGBDFrameDelegate { [weak self] frame in
        self?.send(frame)
    }

    override init() {
        super.init()
        motionQueue.name = "com.visionpipeline.imu.motion"
        motionQueue.maxConcurrentOperationCount = 1
        motionManager.deviceMotionUpdateInterval = 0.01
        arSession.delegate = frameDelegate
        arSession.delegateQueue = frameQueue
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(applicationDidEnterBackground),
            name: UIApplication.didEnterBackgroundNotification,
            object: nil
        )
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(applicationWillEnterForeground),
            name: UIApplication.willEnterForegroundNotification,
            object: nil
        )
        if !ARWorldTrackingConfiguration.isSupported {
            depthCapability = "AR world tracking unavailable"
        } else if !ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            depthCapability = "LiDAR scene depth unavailable"
        } else {
            depthCapability = "Rear LiDAR scene depth available"
        }
    }

    deinit {
        UIApplication.shared.isIdleTimerDisabled = false
        NotificationCenter.default.removeObserver(self)
    }

    func toggleStreaming() {
        isStreaming ? stop() : start()
    }

    func start() {
        guard !isStreaming else { return }
        guard let portNumber = UInt16(port), !host.isEmpty else {
            imuConnectionState = "Enter a valid host and port"
            return
        }
        guard portNumber < UInt16.max else {
            imuConnectionState = "Port must leave port + 1 for RGB-D"
            return
        }
        guard motionManager.isDeviceMotionAvailable else {
            imuConnectionState = "Device motion is unavailable"
            return
        }
        guard ARWorldTrackingConfiguration.isSupported,
              ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) else {
            rgbdConnectionState = "Rear LiDAR scene depth is unavailable"
            return
        }

            sessionID = UUID().uuidString.lowercased()
            samplesSent = 0
            framesSent = 0
            framesDropped = 0
            imuSequenceNumber = 0

        let endpoint = NWEndpoint.Host(host)
        let imuConnection = NWConnection(
            host: endpoint,
            port: NWEndpoint.Port(rawValue: portNumber)!,
            using: .tcp
        )
        let rgbdConnection = NWConnection(
            host: endpoint,
            port: NWEndpoint.Port(rawValue: portNumber + 1)!,
            using: .tcp
        )
        self.imuConnection = imuConnection
        self.rgbdConnection = rgbdConnection
        configureIMUConnection(imuConnection)
        configureRGBDConnection(rgbdConnection)
        imuConnection.start(queue: .global(qos: .userInitiated))
        rgbdConnection.start(queue: .global(qos: .userInitiated))

        motionManager.startDeviceMotionUpdates(using: .xArbitraryZVertical, to: motionQueue) { [weak self] motion, error in
            guard let motion else {
                if let error {
                    DispatchQueue.main.async {
                        self?.imuConnectionState = error.localizedDescription
                    }
                }
                return
            }
            self?.send(motion)
        }

        let configuration = ARWorldTrackingConfiguration()
        configuration.frameSemantics.insert(.sceneDepth)
        configuration.worldAlignment = .gravity
        arSession.run(configuration, options: [.resetTracking, .removeExistingAnchors])

        isStreaming = true
        UIApplication.shared.isIdleTimerDisabled = true
        frameQueue.async { [weak self] in
            self?.frameSequenceNumber = 0
            self?.previousFrameTimestamp = -Double.infinity
            self?.frameSendPending = false
        }
    }

    func stop() {
        wasStreamingBeforeBackground = false
        UIApplication.shared.isIdleTimerDisabled = false
        motionManager.stopDeviceMotionUpdates()
        arSession.pause()
        imuConnection?.cancel()
        rgbdConnection?.cancel()
        imuConnection = nil
        rgbdConnection = nil
        frameQueue.async { [weak self] in
            self?.rgbdConnectionReady = false
            self?.frameSendPending = false
        }
        isStreaming = false
        imuConnectionState = "Not connected"
        rgbdConnectionState = "Not connected"
    }

    @objc private func applicationDidEnterBackground() {
        guard isStreaming else { return }
        wasStreamingBeforeBackground = true
        imuConnectionState = "Paused while locked"
        rgbdConnectionState = "Paused while locked"
    }

    @objc private func applicationWillEnterForeground() {
        guard wasStreamingBeforeBackground else { return }
        wasStreamingBeforeBackground = false
        // Locking interrupts ARKit camera capture, so resume with a new world epoch.
        stop()
        start()
    }

    private func send(_ motion: CMDeviceMotion) {
        let sampleSequenceNumber = imuSequenceNumber
        imuSequenceNumber += 1
        let sample = IMUSample(
            schema: "iphone_imu.v1",
            sourceID: sourceID,
            sessionID: sessionID,
            sequenceNumber: sampleSequenceNumber,
            deviceTimestampNS: UInt64(max(0, motion.timestamp) * 1_000_000_000),
            attitudeQuaternion: Quaternion(
                x: motion.attitude.quaternion.x,
                y: motion.attitude.quaternion.y,
                z: motion.attitude.quaternion.z,
                w: motion.attitude.quaternion.w
            ),
            gravityCompensatedAccelerationMPS2: Vector3(
                x: motion.userAcceleration.x * 9.80665,
                y: motion.userAcceleration.y * 9.80665,
                z: motion.userAcceleration.z * 9.80665
            ),
            angularVelocityRadPS: Vector3(
                x: motion.rotationRate.x,
                y: motion.rotationRate.y,
                z: motion.rotationRate.z
            ),
            gravityMPS2: Vector3(
                x: motion.gravity.x * 9.80665,
                y: motion.gravity.y * 9.80665,
                z: motion.gravity.z * 9.80665
            )
        )
        guard let data = try? encoder.encode(sample) else { return }
        var packet = data
        packet.append(0x0A)
        imuConnection?.send(content: packet, completion: .contentProcessed { [weak self] error in
            guard error == nil else { return }
            DispatchQueue.main.async {
                self?.samplesSent += 1
            }
        })
    }

    private func configureIMUConnection(_ connection: NWConnection) {
        connection.stateUpdateHandler = { [weak self] state in
            DispatchQueue.main.async {
                guard let self else { return }
                switch state {
                case .ready:
                    self.imuConnectionState = "Connected on \(self.port)"
                case .failed(let error):
                    self.imuConnectionState = "Failed: \(error.localizedDescription)"
                case .cancelled:
                    self.imuConnectionState = "Disconnected"
                default:
                    self.imuConnectionState = "Connecting..."
                }
            }
        }
    }

    private func configureRGBDConnection(_ connection: NWConnection) {
        connection.stateUpdateHandler = { [weak self] state in
            guard let self else { return }
            let ready: Bool
            let status: String
            switch state {
            case .ready:
                ready = true
                status = "Connected on port + 1"
            case .failed(let error):
                ready = false
                status = "Failed: \(error.localizedDescription)"
            case .cancelled:
                ready = false
                status = "Disconnected"
            default:
                ready = false
                status = "Connecting..."
            }
            self.frameQueue.async {
                self.rgbdConnectionReady = ready
            }
            DispatchQueue.main.async {
                self.rgbdConnectionState = status
            }
        }
    }

    /// Run on the serial ARSession delegate queue.
    private func send(_ frame: ARFrame) {
        guard rgbdConnectionReady else { return }
        guard !frameSendPending else {
            DispatchQueue.main.async { [weak self] in self?.framesDropped += 1 }
            return
        }
        guard frame.timestamp - previousFrameTimestamp >= frameIntervalSeconds else { return }
        guard let sceneDepth = frame.sceneDepth else { return }

        previousFrameTimestamp = frame.timestamp
        frameSendPending = true

          guard let rgb = encodeJPEG(frame.capturedImage),
              let depth = tightlyPackedBytes(
                sceneDepth.depthMap,
                expectedPixelFormat: kCVPixelFormatType_DepthFloat32,
                bytesPerPixel: 4
              ) else {
            frameSendPending = false
            return
        }
        let confidence = sceneDepth.confidenceMap.flatMap {
            tightlyPackedBytes(
                $0,
                expectedPixelFormat: kCVPixelFormatType_OneComponent8,
                bytesPerPixel: 1
            )
        } ?? Data()
        let rgbWidth = CVPixelBufferGetWidth(frame.capturedImage)
        let rgbHeight = CVPixelBufferGetHeight(frame.capturedImage)
        let depthWidth = CVPixelBufferGetWidth(sceneDepth.depthMap)
        let depthHeight = CVPixelBufferGetHeight(sceneDepth.depthMap)
        let intrinsics = frame.camera.intrinsics
        let transform = frame.camera.transform
        let sequence = frameSequenceNumber
        frameSequenceNumber += 1

        let header = RGBDFrameHeader(
            schema: "iphone_rgbd.v1",
            sourceID: sourceID,
            sessionID: sessionID,
            sequenceNumber: sequence,
            deviceTimestampNS: UInt64(max(0, frame.timestamp) * 1_000_000_000),
            trackingState: trackingStateDescription(frame.camera.trackingState),
            rgbEncoding: "jpeg",
            rgbWidth: rgbWidth,
            rgbHeight: rgbHeight,
            depthEncoding: "float32_le_metres",
            depthWidth: depthWidth,
            depthHeight: depthHeight,
            confidenceEncoding: confidence.isEmpty ? nil : "arkit_uint8_0_low_1_medium_2_high",
            rgbOrientation: "camera_buffer_native",
            cameraIntrinsicsColumnMajor: [
                intrinsics.columns.0.x, intrinsics.columns.0.y, intrinsics.columns.0.z,
                intrinsics.columns.1.x, intrinsics.columns.1.y, intrinsics.columns.1.z,
                intrinsics.columns.2.x, intrinsics.columns.2.y, intrinsics.columns.2.z,
            ],
            cameraTransformColumnMajor: [
                transform.columns.0.x, transform.columns.0.y, transform.columns.0.z, transform.columns.0.w,
                transform.columns.1.x, transform.columns.1.y, transform.columns.1.z, transform.columns.1.w,
                transform.columns.2.x, transform.columns.2.y, transform.columns.2.z, transform.columns.2.w,
                transform.columns.3.x, transform.columns.3.y, transform.columns.3.z, transform.columns.3.w,
            ],
            payload: RGBDPayloadLayout(
                rgbBytes: rgb.count,
                depthBytes: depth.count,
                confidenceBytes: confidence.count
            )
        )

        guard let headerData = try? encoder.encode(header), headerData.count <= Int(UInt32.max) else {
            frameSendPending = false
            return
        }
        var headerSize = UInt32(headerData.count).bigEndian
        var packet = Data(bytes: &headerSize, count: MemoryLayout<UInt32>.size)
        packet.append(headerData)
        packet.append(rgb)
        packet.append(depth)
        packet.append(confidence)

        rgbdConnection?.send(content: packet, completion: .contentProcessed { [weak self] error in
            guard let self else { return }
            self.frameQueue.async {
                self.frameSendPending = false
            }
            guard error == nil else { return }
            DispatchQueue.main.async {
                self.framesSent += 1
            }
        })
    }

    private func encodeJPEG(_ pixelBuffer: CVPixelBuffer) -> Data? {
        let image = CIImage(cvPixelBuffer: pixelBuffer)
        guard let cgImage = ciContext.createCGImage(image, from: image.extent) else { return nil }
        return UIImage(cgImage: cgImage).jpegData(compressionQuality: 0.65)
    }

    private func tightlyPackedBytes(
        _ pixelBuffer: CVPixelBuffer,
        expectedPixelFormat: OSType,
        bytesPerPixel: Int
    ) -> Data? {
        guard CVPixelBufferGetPixelFormatType(pixelBuffer) == expectedPixelFormat else { return nil }
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else { return nil }
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let sourceBytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let packedBytesPerRow = width * bytesPerPixel
        guard sourceBytesPerRow >= packedBytesPerRow else { return nil }

        var output = Data(capacity: packedBytesPerRow * height)
        let bytes = baseAddress.assumingMemoryBound(to: UInt8.self)
        for row in 0..<height {
            output.append(bytes.advanced(by: row * sourceBytesPerRow), count: packedBytesPerRow)
        }
        return output
    }

    private func trackingStateDescription(_ state: ARCamera.TrackingState) -> String {
        switch state {
        case .normal:
            return "normal"
        case .notAvailable:
            return "not_available"
        case .limited(let reason):
            return "limited_\(String(describing: reason))"
        }
    }
}

private final class RGBDFrameDelegate: NSObject, ARSessionDelegate {
    private let onFrame: (ARFrame) -> Void

    init(onFrame: @escaping (ARFrame) -> Void) {
        self.onFrame = onFrame
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        onFrame(frame)
    }
}
