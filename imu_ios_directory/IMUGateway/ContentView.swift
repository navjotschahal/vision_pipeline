import SwiftUI

struct ContentView: View {
    @StateObject private var gateway = SensorGateway()

    var body: some View {
        NavigationStack {
            Form {
                Section("Host receiver") {
                    TextField("Mac IP address", text: $gateway.host)
                        .keyboardType(.numbersAndPunctuation)
                        .textInputAutocapitalization(.never)
                    TextField("IMU TCP port", text: $gateway.port)
                        .keyboardType(.numberPad)
                    Text("RGB-D uses the following port (port + 1).")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }

                Section("Live stream") {
                    LabeledContent("IMU", value: gateway.imuConnectionState)
                    LabeledContent("RGB-D", value: gateway.rgbdConnectionState)
                    LabeledContent("Depth", value: gateway.depthCapability)
                    LabeledContent("IMU samples", value: "\(gateway.samplesSent)")
                    LabeledContent("RGB-D frames", value: "\(gateway.framesSent)")
                    LabeledContent("Frames dropped", value: "\(gateway.framesDropped)")
                    LabeledContent("Rates", value: "IMU 100 Hz / RGB-D 10 Hz")
                    Text("The screen stays awake while streaming.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                    Button(gateway.isStreaming ? "Stop streaming" : "Start streaming") {
                        gateway.toggleStreaming()
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(gateway.isStreaming ? .red : .blue)
                }

                Section {
                    Text("Streams raw Core Motion IMU plus ARKit-synchronized RGB, LiDAR depth, confidence, intrinsics, and visual-inertial camera pose. Rear LiDAR raw infrared imagery is not exposed by the public iOS APIs.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            }
            .navigationTitle("iPhone Sensor Gateway")
        }
        .onDisappear { gateway.stop() }
    }
}

#Preview {
    ContentView()
}
