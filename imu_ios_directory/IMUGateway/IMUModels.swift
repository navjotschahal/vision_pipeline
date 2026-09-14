import Foundation

struct Vector3: Codable {
    let x: Double
    let y: Double
    let z: Double
}

struct Quaternion: Codable {
    let x: Double
    let y: Double
    let z: Double
    let w: Double
}

struct IMUSample: Codable {
    let schema: String
    let sourceID: String
    let sessionID: String
    let sequenceNumber: UInt64
    let deviceTimestampNS: UInt64
    let attitudeQuaternion: Quaternion
    let gravityCompensatedAccelerationMPS2: Vector3
    let angularVelocityRadPS: Vector3
    let gravityMPS2: Vector3

    enum CodingKeys: String, CodingKey {
        case schema
        case sourceID = "source_id"
        case sessionID = "session_id"
        case sequenceNumber = "sequence_number"
        case deviceTimestampNS = "device_timestamp_ns"
        case attitudeQuaternion = "attitude_quaternion"
        case gravityCompensatedAccelerationMPS2 = "gravity_compensated_acceleration_mps2"
        case angularVelocityRadPS = "angular_velocity_radps"
        case gravityMPS2 = "gravity_mps2"
    }
}

struct RGBDPayloadLayout: Codable {
    let rgbBytes: Int
    let depthBytes: Int
    let confidenceBytes: Int

    enum CodingKeys: String, CodingKey {
        case rgbBytes = "rgb_bytes"
        case depthBytes = "depth_bytes"
        case confidenceBytes = "confidence_bytes"
    }
}

/// Header for one length-prefixed RGB-D packet.
///
/// The four-byte, big-endian header length is followed by this JSON header, then
/// JPEG RGB bytes, tightly packed Float32 little-endian depth metres, and the
/// optional UInt8 ARKit depth-confidence plane, in that order.
struct RGBDFrameHeader: Codable {
    let schema: String
    let sourceID: String
    let sessionID: String
    let sequenceNumber: UInt64
    let deviceTimestampNS: UInt64
    let trackingState: String
    let rgbEncoding: String
    let rgbWidth: Int
    let rgbHeight: Int
    let depthEncoding: String
    let depthWidth: Int
    let depthHeight: Int
    let confidenceEncoding: String?
    let rgbOrientation: String
    let cameraIntrinsicsColumnMajor: [Float]
    let cameraTransformColumnMajor: [Float]
    let payload: RGBDPayloadLayout

    enum CodingKeys: String, CodingKey {
        case schema
        case sourceID = "source_id"
        case sessionID = "session_id"
        case sequenceNumber = "sequence_number"
        case deviceTimestampNS = "device_timestamp_ns"
        case trackingState = "tracking_state"
        case rgbEncoding = "rgb_encoding"
        case rgbWidth = "rgb_width"
        case rgbHeight = "rgb_height"
        case depthEncoding = "depth_encoding"
        case depthWidth = "depth_width"
        case depthHeight = "depth_height"
        case confidenceEncoding = "confidence_encoding"
        case rgbOrientation = "rgb_orientation"
        case cameraIntrinsicsColumnMajor = "camera_intrinsics_column_major"
        case cameraTransformColumnMajor = "camera_transform_column_major"
        case payload
    }
}
