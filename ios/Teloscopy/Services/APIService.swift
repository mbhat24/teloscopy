// APIService.swift
// Teloscopy
//
// Network layer for communicating with the Teloscopy analysis server.
// Handles authentication, image uploads, result retrieval, and data sync.
//

import Foundation
import Combine
import UIKit

// MARK: - API Error Types

enum TeloscopyAPIError: LocalizedError {
    case invalidURL
    case invalidResponse
    case httpError(statusCode: Int, message: String)
    case decodingError(Error)
    case networkUnavailable
    case unauthorized
    case serverError(String)
    case uploadFailed(String)
    case timeout
    case unknown(Error)
    
    var errorDescription: String? {
        switch self {
        case .invalidURL:
            return "Invalid server URL. Please check your settings."
        case .invalidResponse:
            return "Received an invalid response from the server."
        case .httpError(let code, let message):
            return "Server error (\(code)): \(message)"
        case .decodingError(let error):
            return "Failed to process server response: \(error.localizedDescription)"
        case .networkUnavailable:
            return "Network is unavailable. Data will be synced when connection is restored."
        case .unauthorized:
            return "Authentication failed. Please log in again."
        case .serverError(let message):
            return "Server error: \(message)"
        case .uploadFailed(let message):
            return "Upload failed: \(message)"
        case .timeout:
            return "Request timed out. Please try again."
        case .unknown(let error):
            return "An unexpected error occurred: \(error.localizedDescription)"
        }
    }
}

// MARK: - API Configuration

struct APIConfiguration {
    static let defaultBaseURL = "http://localhost:5000"
    static let apiVersion = "v1"
    static let requestTimeout: TimeInterval = 30
    static let uploadTimeout: TimeInterval = 120
    
    var baseURL: String {
        let stored = UserDefaults.standard.string(forKey: "server_url") ?? APIConfiguration.defaultBaseURL
        return stored.trimmingCharacters(in: .whitespacesAndNewlines)
    }
    
    var fullBaseURL: String {
        "\(baseURL)/api/\(APIConfiguration.apiVersion)"
    }
}

// MARK: - API Endpoints

enum APIEndpoint {
    case login
    case register
    case refreshToken
    case profile
    case analyses(page: Int, pageSize: Int)
    case analysisDetail(id: String)
    case createAnalysis
    case uploadImage(analysisId: String)
    case startAnalysis(id: String)
    case analysisResult(id: String)
    case analysisHistory
    case longitudinalData
    case syncStatus
    
    var path: String {
        switch self {
        case .login: return "/auth/login"
        case .register: return "/auth/register"
        case .refreshToken: return "/auth/refresh"
        case .profile: return "/user/profile"
        case .analyses(let page, let pageSize): return "/analyses?page=\(page)&page_size=\(pageSize)"
        case .analysisDetail(let id): return "/analyses/\(id)"
        case .createAnalysis: return "/analyses"
        case .uploadImage(let id): return "/analyses/\(id)/images"
        case .startAnalysis(let id): return "/analyses/\(id)/start"
        case .analysisResult(let id): return "/analyses/\(id)/result"
        case .analysisHistory: return "/user/analysis-history"
        case .longitudinalData: return "/user/longitudinal-data"
        case .syncStatus: return "/sync/status"
        }
    }
    
    var method: String {
        switch self {
        case .login, .register, .refreshToken, .createAnalysis, .uploadImage, .startAnalysis:
            return "POST"
        case .profile, .analyses, .analysisDetail, .analysisResult, .analysisHistory,
             .longitudinalData, .syncStatus:
            return "GET"
        }
    }
}

// MARK: - API Service

final class APIService: ObservableObject {
    static let shared = APIService()
    
    @Published var isAuthenticated = false
    @Published var isServerReachable = false
    @Published var currentUser: UserProfile?
    
    private let configuration = APIConfiguration()
    private let session: URLSession
    private var authToken: String?
    private var refreshToken: String?
    private var cancellables = Set<AnyCancellable>()
    
    private let jsonDecoder: JSONDecoder = {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .custom { decoder in
            let container = try decoder.singleValueContainer()
            let dateString = try container.decode(String.self)
            
            // Try ISO8601 first
            let iso8601Formatter = ISO8601DateFormatter()
            iso8601Formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
            if let date = iso8601Formatter.date(from: dateString) {
                return date
            }
            
            // Fallback to standard format
            iso8601Formatter.formatOptions = [.withInternetDateTime]
            if let date = iso8601Formatter.date(from: dateString) {
                return date
            }
            
            // Try common date format
            let dateFormatter = DateFormatter()
            dateFormatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ss"
            if let date = dateFormatter.date(from: dateString) {
                return date
            }
            
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "Cannot decode date: \(dateString)"
            )
        }
        return decoder
    }()
    
    private let jsonEncoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }()
    
    private init() {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = APIConfiguration.requestTimeout
        config.timeoutIntervalForResource = APIConfiguration.uploadTimeout
        config.waitsForConnectivity = true
        config.allowsCellularAccess = true
        session = URLSession(configuration: config)
        
        // Restore auth state
        restoreAuthState()
    }
    
    // MARK: - Authentication
    
    func login(username: String, password: String) -> AnyPublisher<UserProfile, TeloscopyAPIError> {
        let body: [String: Any] = [
            "username": username,
            "password": password
        ]
        
        return request(endpoint: .login, body: body)
            .decode(type: LoginResponse.self, decoder: jsonDecoder)
            .mapError { error -> TeloscopyAPIError in
                if let apiError = error as? TeloscopyAPIError {
                    return apiError
                }
                return .decodingError(error)
            }
            .handleEvents(receiveOutput: { [weak self] response in
                self?.handleLoginResponse(response)
            })
            .map(\.user)
            .eraseToAnyPublisher()
    }
    
    func logout() {
        authToken = nil
        refreshToken = nil
        currentUser = nil
        isAuthenticated = false
        
        // Clear stored credentials
        UserDefaults.standard.removeObject(forKey: "auth_token")
        UserDefaults.standard.removeObject(forKey: "refresh_token")
        UserDefaults.standard.removeObject(forKey: "current_user")
    }
    
    func register(username: String, email: String, password: String, fullName: String) -> AnyPublisher<UserProfile, TeloscopyAPIError> {
        let body: [String: Any] = [
            "username": username,
            "email": email,
            "password": password,
            "full_name": fullName
        ]
        
        return request(endpoint: .register, body: body)
            .decode(type: LoginResponse.self, decoder: jsonDecoder)
            .mapError { error -> TeloscopyAPIError in
                if let apiError = error as? TeloscopyAPIError {
                    return apiError
                }
                return .decodingError(error)
            }
            .handleEvents(receiveOutput: { [weak self] response in
                self?.handleLoginResponse(response)
            })
            .map(\.user)
            .eraseToAnyPublisher()
    }
    
    // MARK: - Analysis Operations
    
    func fetchAnalyses(page: Int = 1, pageSize: Int = 20) -> AnyPublisher<[Analysis], TeloscopyAPIError> {
        return request(endpoint: .analyses(page: page, pageSize: pageSize))
            .decode(type: AnalysisListResponse.self, decoder: jsonDecoder)
            .mapError { error -> TeloscopyAPIError in
                if let apiError = error as? TeloscopyAPIError {
                    return apiError
                }
                return .decodingError(error)
            }
            .map(\.analyses)
            .eraseToAnyPublisher()
    }
    
    func fetchAnalysisDetail(id: String) -> AnyPublisher<Analysis, TeloscopyAPIError> {
        return request(endpoint: .analysisDetail(id: id))
            .decode(type: Analysis.self, decoder: jsonDecoder)
            .mapError { error -> TeloscopyAPIError in
                if let apiError = error as? TeloscopyAPIError {
                    return apiError
                }
                return .decodingError(error)
            }
            .eraseToAnyPublisher()
    }
    
    func createAnalysis(name: String, type: AnalysisType, sampleId: String?, patientId: String?, notes: String?) -> AnyPublisher<Analysis, TeloscopyAPIError> {
        var body: [String: Any] = [
            "name": name,
            "analysis_type": type.rawValue
        ]
        if let sampleId = sampleId { body["sample_id"] = sampleId }
        if let patientId = patientId { body["patient_id"] = patientId }
        if let notes = notes { body["notes"] = notes }
        
        return request(endpoint: .createAnalysis, body: body)
            .decode(type: Analysis.self, decoder: jsonDecoder)
            .mapError { error -> TeloscopyAPIError in
                if let apiError = error as? TeloscopyAPIError {
                    return apiError
                }
                return .decodingError(error)
            }
            .eraseToAnyPublisher()
    }
    
    func uploadImage(analysisId: String, imageData: Data, fileName: String) -> AnyPublisher<UploadResponse, TeloscopyAPIError> {
        let urlString = "\(configuration.fullBaseURL)\(APIEndpoint.uploadImage(analysisId: analysisId).path)"
        
        guard let url = URL(string: urlString) else {
            return Fail(error: TeloscopyAPIError.invalidURL).eraseToAnyPublisher()
        }
        
        let boundary = UUID().uuidString
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        
        if let token = authToken {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        
        // Build multipart form data
        var body = Data()
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"image\"; filename=\"\(fileName)\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
        body.append(imageData)
        body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)
        
        request.httpBody = body
        request.timeoutInterval = APIConfiguration.uploadTimeout
        
        return session.dataTaskPublisher(for: request)
            .tryMap { [weak self] data, response -> Data in
                guard let httpResponse = response as? HTTPURLResponse else {
                    throw TeloscopyAPIError.invalidResponse
                }
                
                if httpResponse.statusCode == 401 {
                    self?.handleUnauthorized()
                    throw TeloscopyAPIError.unauthorized
                }
                
                guard (200...299).contains(httpResponse.statusCode) else {
                    let message = String(data: data, encoding: .utf8) ?? "Upload failed"
                    throw TeloscopyAPIError.uploadFailed(message)
                }
                
                return data
            }
            .decode(type: UploadResponse.self, decoder: jsonDecoder)
            .mapError { error -> TeloscopyAPIError in
                if let apiError = error as? TeloscopyAPIError {
                    return apiError
                }
                if let urlError = error as? URLError {
                    switch urlError.code {
                    case .notConnectedToInternet, .networkConnectionLost:
                        return .networkUnavailable
                    case .timedOut:
                        return .timeout
                    default:
                        return .unknown(urlError)
                    }
                }
                return .decodingError(error)
            }
            .eraseToAnyPublisher()
    }
    
    func startAnalysis(id: String) -> AnyPublisher<Analysis, TeloscopyAPIError> {
        return request(endpoint: .startAnalysis(id: id), body: [:])
            .decode(type: Analysis.self, decoder: jsonDecoder)
            .mapError { error -> TeloscopyAPIError in
                if let apiError = error as? TeloscopyAPIError {
                    return apiError
                }
                return .decodingError(error)
            }
            .eraseToAnyPublisher()
    }
    
    func fetchResult(analysisId: String) -> AnyPublisher<TelomereResult, TeloscopyAPIError> {
        return request(endpoint: .analysisResult(id: analysisId))
            .decode(type: TelomereResult.self, decoder: jsonDecoder)
            .mapError { error -> TeloscopyAPIError in
                if let apiError = error as? TeloscopyAPIError {
                    return apiError
                }
                return .decodingError(error)
            }
            .eraseToAnyPublisher()
    }
    
    // MARK: - User & Profile
    
    func fetchProfile() -> AnyPublisher<UserProfile, TeloscopyAPIError> {
        return request(endpoint: .profile)
            .decode(type: UserProfile.self, decoder: jsonDecoder)
            .mapError { error -> TeloscopyAPIError in
                if let apiError = error as? TeloscopyAPIError {
                    return apiError
                }
                return .decodingError(error)
            }
            .handleEvents(receiveOutput: { [weak self] profile in
                DispatchQueue.main.async {
                    self?.currentUser = profile
                }
            })
            .eraseToAnyPublisher()
    }
    
    func fetchLongitudinalData() -> AnyPublisher<[LongitudinalDataPoint], TeloscopyAPIError> {
        return request(endpoint: .longitudinalData)
            .decode(type: [LongitudinalDataPoint].self, decoder: jsonDecoder)
            .mapError { error -> TeloscopyAPIError in
                if let apiError = error as? TeloscopyAPIError {
                    return apiError
                }
                return .decodingError(error)
            }
            .eraseToAnyPublisher()
    }
    
    // MARK: - Server Health
    
    func checkServerHealth() -> AnyPublisher<Bool, Never> {
        let urlString = "\(configuration.baseURL)/health"
        guard let url = URL(string: urlString) else {
            return Just(false).eraseToAnyPublisher()
        }
        
        var request = URLRequest(url: url)
        request.timeoutInterval = 10
        
        return session.dataTaskPublisher(for: request)
            .map { _, response in
                guard let httpResponse = response as? HTTPURLResponse else { return false }
                return (200...299).contains(httpResponse.statusCode)
            }
            .replaceError(with: false)
            .handleEvents(receiveOutput: { [weak self] reachable in
                DispatchQueue.main.async {
                    self?.isServerReachable = reachable
                }
            })
            .eraseToAnyPublisher()
    }
    
    // MARK: - Private Helpers
    
    private func request(endpoint: APIEndpoint, body: [String: Any]? = nil) -> AnyPublisher<Data, TeloscopyAPIError> {
        let urlString = "\(configuration.fullBaseURL)\(endpoint.path)"
        
        guard let url = URL(string: urlString) else {
            return Fail(error: TeloscopyAPIError.invalidURL).eraseToAnyPublisher()
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = endpoint.method
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.setValue("Teloscopy-iOS/1.0.0", forHTTPHeaderField: "User-Agent")
        
        if let token = authToken {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        
        if let body = body, endpoint.method == "POST" || endpoint.method == "PUT" {
            request.httpBody = try? JSONSerialization.data(withJSONObject: body)
        }
        
        return session.dataTaskPublisher(for: request)
            .tryMap { [weak self] data, response -> Data in
                guard let httpResponse = response as? HTTPURLResponse else {
                    throw TeloscopyAPIError.invalidResponse
                }
                
                switch httpResponse.statusCode {
                case 200...299:
                    return data
                case 401:
                    self?.handleUnauthorized()
                    throw TeloscopyAPIError.unauthorized
                case 400...499:
                    let message = String(data: data, encoding: .utf8) ?? "Client error"
                    throw TeloscopyAPIError.httpError(statusCode: httpResponse.statusCode, message: message)
                case 500...599:
                    let message = String(data: data, encoding: .utf8) ?? "Server error"
                    throw TeloscopyAPIError.serverError(message)
                default:
                    throw TeloscopyAPIError.httpError(statusCode: httpResponse.statusCode, message: "Unexpected status")
                }
            }
            .mapError { error -> TeloscopyAPIError in
                if let apiError = error as? TeloscopyAPIError {
                    return apiError
                }
                if let urlError = error as? URLError {
                    switch urlError.code {
                    case .notConnectedToInternet, .networkConnectionLost:
                        return .networkUnavailable
                    case .timedOut:
                        return .timeout
                    default:
                        return .unknown(urlError)
                    }
                }
                return .unknown(error)
            }
            .eraseToAnyPublisher()
    }
    
    private func handleLoginResponse(_ response: LoginResponse) {
        DispatchQueue.main.async { [weak self] in
            self?.authToken = response.token
            self?.refreshToken = response.refreshToken
            self?.currentUser = response.user
            self?.isAuthenticated = true
            
            // Persist auth state
            UserDefaults.standard.set(response.token, forKey: "auth_token")
            if let refreshToken = response.refreshToken {
                UserDefaults.standard.set(refreshToken, forKey: "refresh_token")
            }
            if let userData = try? JSONEncoder().encode(response.user) {
                UserDefaults.standard.set(userData, forKey: "current_user")
            }
        }
    }
    
    private func restoreAuthState() {
        if let token = UserDefaults.standard.string(forKey: "auth_token") {
            authToken = token
            refreshToken = UserDefaults.standard.string(forKey: "refresh_token")
            
            if let userData = UserDefaults.standard.data(forKey: "current_user"),
               let user = try? JSONDecoder().decode(UserProfile.self, from: userData) {
                currentUser = user
                isAuthenticated = true
            }
        }
    }
    
    // MARK: - Async/Await Profile Analysis Methods

    func profileAnalysis(request: ProfileAnalysisRequest) async throws -> ProfileAnalysisResponse {
        let body = try jsonEncoder.encode(request)
        return try await performAsyncRequest(
            path: "/profile/analyze",
            method: "POST",
            body: body,
            responseType: ProfileAnalysisResponse.self
        )
    }

    func diseaseRisk(request: DiseaseRiskRequest) async throws -> DiseaseRiskResponse {
        let body = try jsonEncoder.encode(request)
        return try await performAsyncRequest(
            path: "/profile/disease-risk",
            method: "POST",
            body: body,
            responseType: DiseaseRiskResponse.self
        )
    }

    func nutrition(request: NutritionRequest) async throws -> NutritionResponse {
        let body = try jsonEncoder.encode(request)
        return try await performAsyncRequest(
            path: "/profile/nutrition",
            method: "POST",
            body: body,
            responseType: NutritionResponse.self
        )
    }

    private func performAsyncRequest<T: Decodable>(
        path: String,
        method: String,
        body: Data?,
        responseType: T.Type
    ) async throws -> T {
        let urlString = "\(configuration.fullBaseURL)\(path)"
        guard let url = URL(string: urlString) else {
            throw TeloscopyAPIError.invalidURL
        }

        var request = URLRequest(url: url)
        request.httpMethod = method
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.setValue("Teloscopy-iOS/1.0.0", forHTTPHeaderField: "User-Agent")

        if let token = authToken {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }

        request.httpBody = body

        let (data, response) = try await session.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw TeloscopyAPIError.invalidResponse
        }

        switch httpResponse.statusCode {
        case 200...299:
            return try jsonDecoder.decode(T.self, from: data)
        case 401:
            handleUnauthorized()
            throw TeloscopyAPIError.unauthorized
        case 400...499:
            let message = String(data: data, encoding: .utf8) ?? "Client error"
            throw TeloscopyAPIError.httpError(statusCode: httpResponse.statusCode, message: message)
        case 500...599:
            let message = String(data: data, encoding: .utf8) ?? "Server error"
            throw TeloscopyAPIError.serverError(message)
        default:
            throw TeloscopyAPIError.httpError(statusCode: httpResponse.statusCode, message: "Unexpected status")
        }
    }

    private func handleUnauthorized() {
        DispatchQueue.main.async { [weak self] in
            self?.isAuthenticated = false
            self?.authToken = nil
            UserDefaults.standard.removeObject(forKey: "auth_token")
        }
    }
}
