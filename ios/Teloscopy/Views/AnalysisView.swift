// AnalysisView.swift
// Teloscopy
//
// Start new analysis: choose type, upload images via camera/photo library,
// configure parameters, and submit for processing.
//

import SwiftUI
import PhotosUI
import Combine

struct AnalysisView: View {
    @EnvironmentObject var apiService: APIService
    @EnvironmentObject var syncManager: SyncManager
    
    @State private var analysisName = ""
    @State private var selectedType: AnalysisType = .telomereLength
    @State private var sampleId = ""
    @State private var patientId = ""
    @State private var notes = ""
    @State private var selectedImages: [UIImage] = []
    @State private var showImagePicker = false
    @State private var showCamera = false
    @State private var showSourcePicker = false
    @State private var isSubmitting = false
    @State private var showSuccessAlert = false
    @State private var showErrorAlert = false
    @State private var errorMessage = ""
    @State private var cancellables = Set<AnyCancellable>()
    @State private var selectedPhotoItems: [PhotosPickerItem] = []
    
    var body: some View {
        ScrollView {
            VStack(spacing: 24) {
                analysisTypeSection
                detailsSection
                imageSection
                parametersSection
                submitSection
            }
            .padding()
        }
        .background(TeloscopyTheme.surfaceBackground.ignoresSafeArea())
        .navigationTitle("New Analysis")
        .navigationBarTitleDisplayMode(.large)
        .photosPicker(isPresented: $showImagePicker, selection: $selectedPhotoItems, maxSelectionCount: 20, matching: .images)
        .onChange(of: selectedPhotoItems) { newItems in
            loadSelectedPhotos(newItems)
        }
        .sheet(isPresented: $showCamera) {
            CameraView { image in
                if let image = image {
                    selectedImages.append(image)
                }
            }
        }
        .confirmationDialog("Add Images", isPresented: $showSourcePicker) {
            Button("Camera") { showCamera = true }
            Button("Photo Library") { showImagePicker = true }
            Button("Cancel", role: .cancel) { }
        } message: {
            Text("Choose image source for telomere analysis")
        }
        .alert("Analysis Submitted", isPresented: $showSuccessAlert) {
            Button("OK") { resetForm() }
        } message: {
            Text("Your analysis has been queued for processing. You'll be notified when results are ready.")
        }
        .alert("Error", isPresented: $showErrorAlert) {
            Button("OK") { }
        } message: {
            Text(errorMessage)
        }
    }
    
    // MARK: - Analysis Type Selection
    
    private var analysisTypeSection: some View {
        VStack(alignment: .leading, spacing: 14) {
            Label("Analysis Type", systemImage: "flask")
                .font(.headline)
                .foregroundColor(TeloscopyTheme.textPrimary)
            
            ForEach(AnalysisType.allCases) { type in
                AnalysisTypeCard(
                    type: type,
                    isSelected: selectedType == type
                ) {
                    withAnimation(.spring(response: 0.3)) {
                        selectedType = type
                    }
                }
            }
        }
    }
    
    // MARK: - Details
    
    private var detailsSection: some View {
        VStack(alignment: .leading, spacing: 14) {
            Label("Analysis Details", systemImage: "doc.text")
                .font(.headline)
                .foregroundColor(TeloscopyTheme.textPrimary)
            
            VStack(spacing: 16) {
                FormTextField(
                    title: "Analysis Name",
                    placeholder: "e.g., TL-2024-042",
                    text: $analysisName,
                    icon: "tag"
                )
                
                FormTextField(
                    title: "Sample ID",
                    placeholder: "e.g., BLD-001",
                    text: $sampleId,
                    icon: "testtube.2"
                )
                
                FormTextField(
                    title: "Patient ID (Optional)",
                    placeholder: "e.g., PT-1001",
                    text: $patientId,
                    icon: "person"
                )
                
                VStack(alignment: .leading, spacing: 6) {
                    Label("Notes", systemImage: "note.text")
                        .font(.caption)
                        .fontWeight(.medium)
                        .foregroundColor(TeloscopyTheme.textSecondary)
                    
                    TextEditor(text: $notes)
                        .frame(minHeight: 80)
                        .padding(10)
                        .background(TeloscopyTheme.surfaceBackground)
                        .cornerRadius(TeloscopyTheme.smallCornerRadius)
                        .overlay(
                            RoundedRectangle(cornerRadius: TeloscopyTheme.smallCornerRadius)
                                .stroke(Color.gray.opacity(0.2), lineWidth: 1)
                        )
                }
            }
            .padding()
            .cardStyle()
        }
    }
    
    // MARK: - Image Section
    
    private var imageSection: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Label("Microscope Images", systemImage: "photo.on.rectangle.angled")
                    .font(.headline)
                    .foregroundColor(TeloscopyTheme.textPrimary)
                
                Spacer()
                
                Text("\(selectedImages.count) selected")
                    .font(.caption)
                    .foregroundColor(TeloscopyTheme.textSecondary)
            }
            
            // Image grid
            if selectedImages.isEmpty {
                emptyImagePrompt
            } else {
                imageGrid
            }
            
            // Add more images button
            Button(action: { showSourcePicker = true }) {
                HStack {
                    Image(systemName: "plus.circle.fill")
                    Text("Add Images")
                        .fontWeight(.medium)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 14)
                .background(TeloscopyTheme.primaryBlue.opacity(0.1))
                .foregroundColor(TeloscopyTheme.primaryBlue)
                .cornerRadius(TeloscopyTheme.smallCornerRadius)
            }
        }
    }
    
    private var emptyImagePrompt: some View {
        VStack(spacing: 16) {
            Image(systemName: "camera.viewfinder")
                .font(.system(size: 44))
                .foregroundColor(TeloscopyTheme.primaryBlue.opacity(0.4))
            
            Text("No images added yet")
                .font(.subheadline)
                .foregroundColor(TeloscopyTheme.textSecondary)
            
            Text("Capture or select microscope images of chromosomes for telomere analysis")
                .font(.caption)
                .foregroundColor(TeloscopyTheme.textSecondary)
                .multilineTextAlignment(.center)
            
            HStack(spacing: 12) {
                Button(action: { showCamera = true }) {
                    Label("Camera", systemImage: "camera.fill")
                        .font(.subheadline)
                        .fontWeight(.medium)
                        .padding(.horizontal, 20)
                        .padding(.vertical, 12)
                        .background(TeloscopyTheme.primaryBlue)
                        .foregroundColor(.white)
                        .cornerRadius(TeloscopyTheme.smallCornerRadius)
                }
                
                Button(action: { showImagePicker = true }) {
                    Label("Library", systemImage: "photo.on.rectangle")
                        .font(.subheadline)
                        .fontWeight(.medium)
                        .padding(.horizontal, 20)
                        .padding(.vertical, 12)
                        .background(TeloscopyTheme.surfaceBackground)
                        .foregroundColor(TeloscopyTheme.primaryBlue)
                        .cornerRadius(TeloscopyTheme.smallCornerRadius)
                        .overlay(
                            RoundedRectangle(cornerRadius: TeloscopyTheme.smallCornerRadius)
                                .stroke(TeloscopyTheme.primaryBlue, lineWidth: 1)
                        )
                }
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 30)
        .padding(.horizontal)
        .cardStyle()
    }
    
    private var imageGrid: some View {
        let columns = [
            GridItem(.adaptive(minimum: 80, maximum: 100), spacing: 8)
        ]
        
        return LazyVGrid(columns: columns, spacing: 8) {
            ForEach(selectedImages.indices, id: \.self) { index in
                ZStack(alignment: .topTrailing) {
                    Image(uiImage: selectedImages[index])
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                        .frame(width: 90, height: 90)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                    
                    Button(action: {
                        withAnimation {
                            selectedImages.remove(at: index)
                        }
                    }) {
                        Image(systemName: "xmark.circle.fill")
                            .font(.system(size: 20))
                            .foregroundColor(.white)
                            .shadow(radius: 2)
                    }
                    .offset(x: 4, y: -4)
                }
            }
        }
        .padding()
        .cardStyle()
    }
    
    // MARK: - Parameters
    
    private var parametersSection: some View {
        VStack(alignment: .leading, spacing: 14) {
            Label("Quality Settings", systemImage: "slider.horizontal.3")
                .font(.headline)
                .foregroundColor(TeloscopyTheme.textPrimary)
            
            VStack(spacing: 16) {
                HStack {
                    Image(systemName: "checkmark.shield.fill")
                        .foregroundColor(TeloscopyTheme.successGreen)
                    
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Auto Quality Check")
                            .font(.subheadline)
                            .fontWeight(.medium)
                        Text("Images will be validated before analysis")
                            .font(.caption)
                            .foregroundColor(TeloscopyTheme.textSecondary)
                    }
                    
                    Spacer()
                    
                    Image(systemName: "checkmark")
                        .foregroundColor(TeloscopyTheme.successGreen)
                }
                
                Divider()
                
                HStack {
                    Image(systemName: "cpu")
                        .foregroundColor(TeloscopyTheme.primaryBlue)
                    
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Processing Mode")
                            .font(.subheadline)
                            .fontWeight(.medium)
                        Text("Standard analysis with full chromosome mapping")
                            .font(.caption)
                            .foregroundColor(TeloscopyTheme.textSecondary)
                    }
                    
                    Spacer()
                }
            }
            .padding()
            .cardStyle()
        }
    }
    
    // MARK: - Submit
    
    private var submitSection: some View {
        VStack(spacing: 12) {
            Button(action: submitAnalysis) {
                HStack(spacing: 10) {
                    if isSubmitting {
                        ProgressView()
                            .progressViewStyle(CircularProgressViewStyle(tint: .white))
                            .scaleEffect(0.9)
                    } else {
                        Image(systemName: "play.fill")
                    }
                    
                    Text(isSubmitting ? "Submitting..." : "Start Analysis")
                        .fontWeight(.semibold)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 16)
                .background(
                    LinearGradient(
                        colors: canSubmit
                            ? [TeloscopyTheme.primaryBlue, TeloscopyTheme.darkBlue]
                            : [Color.gray.opacity(0.4), Color.gray.opacity(0.3)],
                        startPoint: .leading,
                        endPoint: .trailing
                    )
                )
                .foregroundColor(.white)
                .cornerRadius(TeloscopyTheme.cornerRadius)
            }
            .disabled(!canSubmit || isSubmitting)
            
            if !canSubmit {
                Text(validationMessage)
                    .font(.caption)
                    .foregroundColor(TeloscopyTheme.textSecondary)
                    .multilineTextAlignment(.center)
            }
        }
        .padding(.bottom, 20)
    }
    
    // MARK: - Helpers
    
    private var canSubmit: Bool {
        !analysisName.isEmpty && !selectedImages.isEmpty
    }
    
    private var validationMessage: String {
        if analysisName.isEmpty && selectedImages.isEmpty {
            return "Please enter an analysis name and add at least one image"
        } else if analysisName.isEmpty {
            return "Please enter an analysis name"
        } else if selectedImages.isEmpty {
            return "Please add at least one microscope image"
        }
        return ""
    }
    
    private func loadSelectedPhotos(_ items: [PhotosPickerItem]) {
        for item in items {
            item.loadTransferable(type: Data.self) { result in
                switch result {
                case .success(let data):
                    if let data = data, let image = UIImage(data: data) {
                        DispatchQueue.main.async {
                            selectedImages.append(image)
                        }
                    }
                case .failure(let error):
                    print("[AnalysisView] Failed to load photo: \(error)")
                }
            }
        }
    }
    
    private func submitAnalysis() {
        isSubmitting = true
        
        // Create local analysis first
        let analysis = Analysis(
            name: analysisName,
            analysisType: selectedType,
            status: .uploading,
            sampleId: sampleId.isEmpty ? nil : sampleId,
            patientId: patientId.isEmpty ? nil : patientId,
            notes: notes.isEmpty ? nil : notes,
            imageCount: selectedImages.count
        )
        
        // Save locally for offline support
        syncManager.saveAnalysisLocally(analysis)
        
        // Save images locally and queue uploads
        for (index, image) in selectedImages.enumerated() {
            if let imageData = image.jpegData(compressionQuality: 0.85) {
                let fileName = "\(analysis.id.uuidString)_\(index).jpg"
                if let localPath = syncManager.saveImageLocally(imageData, fileName: fileName) {
                    syncManager.queueUpload(
                        analysisId: analysis.id,
                        localImagePath: localPath,
                        fileName: fileName
                    )
                }
            }
        }
        
        // Try to create on server
        if syncManager.isNetworkAvailable {
            apiService.createAnalysis(
                name: analysisName,
                type: selectedType,
                sampleId: sampleId.isEmpty ? nil : sampleId,
                patientId: patientId.isEmpty ? nil : patientId,
                notes: notes.isEmpty ? nil : notes
            )
            .receive(on: DispatchQueue.main)
            .sink(
                receiveCompletion: { [self] completion in
                    isSubmitting = false
                    switch completion {
                    case .finished:
                        showSuccessAlert = true
                    case .failure(let error):
                        // Already saved locally, show warning but not failure
                        showSuccessAlert = true
                        print("[AnalysisView] Server create failed, queued locally: \(error)")
                    }
                },
                receiveValue: { _ in }
            )
            .store(in: &cancellables)
        } else {
            isSubmitting = false
            showSuccessAlert = true
        }
    }
    
    private func resetForm() {
        analysisName = ""
        selectedType = .telomereLength
        sampleId = ""
        patientId = ""
        notes = ""
        selectedImages = []
        selectedPhotoItems = []
    }
}

// MARK: - Analysis Type Card

struct AnalysisTypeCard: View {
    let type: AnalysisType
    let isSelected: Bool
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            HStack(spacing: 14) {
                ZStack {
                    Circle()
                        .fill(isSelected ? TeloscopyTheme.primaryBlue.opacity(0.15) : TeloscopyTheme.surfaceBackground)
                        .frame(width: 44, height: 44)
                    
                    Image(systemName: type.iconName)
                        .font(.system(size: 18))
                        .foregroundColor(isSelected ? TeloscopyTheme.primaryBlue : TeloscopyTheme.textSecondary)
                }
                
                VStack(alignment: .leading, spacing: 3) {
                    Text(type.displayName)
                        .font(.subheadline)
                        .fontWeight(.semibold)
                        .foregroundColor(TeloscopyTheme.textPrimary)
                    
                    Text(type.description)
                        .font(.caption2)
                        .foregroundColor(TeloscopyTheme.textSecondary)
                        .lineLimit(2)
                }
                
                Spacer()
                
                Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                    .font(.title3)
                    .foregroundColor(isSelected ? TeloscopyTheme.primaryBlue : Color.gray.opacity(0.3))
            }
            .padding()
            .background(isSelected ? TeloscopyTheme.primaryBlue.opacity(0.05) : TeloscopyTheme.cardBackground)
            .cornerRadius(TeloscopyTheme.cornerRadius)
            .overlay(
                RoundedRectangle(cornerRadius: TeloscopyTheme.cornerRadius)
                    .stroke(isSelected ? TeloscopyTheme.primaryBlue : Color.clear, lineWidth: 1.5)
            )
            .shadow(color: TeloscopyTheme.cardShadow, radius: isSelected ? 4 : 2, x: 0, y: 1)
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Form Text Field

struct FormTextField: View {
    let title: String
    let placeholder: String
    @Binding var text: String
    let icon: String
    
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Label(title, systemImage: icon)
                .font(.caption)
                .fontWeight(.medium)
                .foregroundColor(TeloscopyTheme.textSecondary)
            
            TextField(placeholder, text: $text)
                .textFieldStyle(.plain)
                .padding(12)
                .background(TeloscopyTheme.surfaceBackground)
                .cornerRadius(TeloscopyTheme.smallCornerRadius)
                .overlay(
                    RoundedRectangle(cornerRadius: TeloscopyTheme.smallCornerRadius)
                        .stroke(Color.gray.opacity(0.15), lineWidth: 1)
                )
        }
    }
}

// MARK: - Camera View (UIViewControllerRepresentable)

struct CameraView: UIViewControllerRepresentable {
    let onCapture: (UIImage?) -> Void
    @Environment(\.dismiss) var dismiss
    
    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.sourceType = .camera
        picker.delegate = context.coordinator
        picker.allowsEditing = false
        return picker
    }
    
    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}
    
    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }
    
    class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        let parent: CameraView
        
        init(_ parent: CameraView) {
            self.parent = parent
        }
        
        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]) {
            let image = info[.originalImage] as? UIImage
            parent.onCapture(image)
            parent.dismiss()
        }
        
        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            parent.onCapture(nil)
            parent.dismiss()
        }
    }
}

// MARK: - Preview

#Preview {
    NavigationStack {
        AnalysisView()
    }
    .environmentObject(APIService.shared)
    .environmentObject(SyncManager.shared)
}
