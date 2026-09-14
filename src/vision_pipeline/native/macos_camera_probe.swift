import AVFoundation
import CoreMedia
import Foundation

struct CameraProfile: Codable {
    let width: Int32
    let height: Int32
    let min_fps: Double
    let max_fps: Double
    let pixel_format: String
}

struct CameraDevice: Codable {
    let index: Int
    let name: String
    let unique_id: String
    let position: String
    let is_connected: Bool
    let is_suspended: Bool
    let profiles: [CameraProfile]
}

func fourCC(_ value: FourCharCode) -> String {
    let bytes: [UInt8] = [
        UInt8((value >> 24) & 0xff),
        UInt8((value >> 16) & 0xff),
        UInt8((value >> 8) & 0xff),
        UInt8(value & 0xff),
    ]
    if bytes.allSatisfy({ $0 >= 32 && $0 <= 126 }) {
        return String(bytes: bytes, encoding: .ascii) ?? String(format: "0x%08x", value)
    }
    return String(format: "0x%08x", value)
}

func positionName(_ position: AVCaptureDevice.Position) -> String {
    switch position {
    case .front:
        return "front"
    case .back:
        return "back"
    case .unspecified:
        return "unspecified"
    @unknown default:
        return "unknown"
    }
}

let discovery = AVCaptureDevice.DiscoverySession(
    deviceTypes: [.builtInWideAngleCamera, .continuityCamera, .external],
    mediaType: .video,
    position: .unspecified
)
let devices = discovery.devices
let result = devices.enumerated().map { index, device in
    let profiles = device.formats.flatMap { format in
        let dimensions = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
        let pixelFormat = fourCC(CMFormatDescriptionGetMediaSubType(format.formatDescription))
        return format.videoSupportedFrameRateRanges.map { range in
            CameraProfile(
                width: dimensions.width,
                height: dimensions.height,
                min_fps: range.minFrameRate,
                max_fps: range.maxFrameRate,
                pixel_format: pixelFormat
            )
        }
    }
    return CameraDevice(
        index: index,
        name: device.localizedName,
        unique_id: device.uniqueID,
        position: positionName(device.position),
        is_connected: device.isConnected,
        is_suspended: device.isSuspended,
        profiles: profiles
    )
}

let encoder = JSONEncoder()
encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
let data = try encoder.encode(result)
FileHandle.standardOutput.write(data)
FileHandle.standardOutput.write(Data("\n".utf8))
