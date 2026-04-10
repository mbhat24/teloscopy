plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
    alias(libs.plugins.hilt)
    alias(libs.plugins.ksp)
}

android {
    namespace = "com.teloscopy.app"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.teloscopy.app"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "2.0.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        vectorDrawables {
            useSupportLibrary = true
        }
    }

    // Release signing configuration
    // To sign release builds, create a keystore and uncomment the following block.
    // Store credentials in ~/.gradle/gradle.properties or use environment variables:
    //
    //   TELOSCOPY_KEYSTORE_FILE=/path/to/teloscopy-release.jks
    //   TELOSCOPY_KEYSTORE_PASSWORD=your_keystore_password
    //   TELOSCOPY_KEY_ALIAS=teloscopy
    //   TELOSCOPY_KEY_PASSWORD=your_key_password
    //
    // signingConfigs {
    //     create("release") {
    //         storeFile = file(System.getenv("TELOSCOPY_KEYSTORE_FILE") ?: "teloscopy-release.jks")
    //         storePassword = System.getenv("TELOSCOPY_KEYSTORE_PASSWORD") ?: ""
    //         keyAlias = System.getenv("TELOSCOPY_KEY_ALIAS") ?: "teloscopy"
    //         keyPassword = System.getenv("TELOSCOPY_KEY_PASSWORD") ?: ""
    //     }
    // }

    buildTypes {
        debug {
            isDebuggable = true
            applicationIdSuffix = ".debug"
            versionNameSuffix = "-debug"
        }
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
            // Uncomment after configuring signingConfigs above:
            // signingConfig = signingConfigs.getByName("release")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    buildFeatures {
        compose = true
        buildConfig = true
    }

    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }
}

dependencies {
    // Compose BOM - manages all Compose library versions
    val composeBom = platform(libs.compose.bom)
    implementation(composeBom)
    androidTestImplementation(composeBom)

    // Compose UI
    implementation(libs.compose.ui)
    implementation(libs.compose.ui.graphics)
    implementation(libs.compose.ui.tooling.preview)
    implementation(libs.compose.material3)
    implementation(libs.compose.material.icons.extended)
    debugImplementation(libs.compose.ui.tooling)
    debugImplementation(libs.compose.ui.test.manifest)

    // Navigation
    implementation(libs.navigation.compose)

    // Hilt - Dependency Injection
    implementation(libs.hilt.android)
    ksp(libs.hilt.android.compiler)
    implementation(libs.hilt.navigation.compose)

    // Retrofit & Networking
    implementation(libs.retrofit)
    implementation(libs.retrofit.converter.moshi)
    implementation(libs.okhttp)
    implementation(libs.okhttp.logging.interceptor)

    // Moshi - JSON parsing
    implementation(libs.moshi)
    implementation(libs.moshi.kotlin)
    ksp(libs.moshi.kotlin.codegen)

    // CameraX - barcode/sample scanning
    implementation(libs.camerax.core)
    implementation(libs.camerax.camera2)
    implementation(libs.camerax.lifecycle)
    implementation(libs.camerax.view)

    // Coil - image loading
    implementation(libs.coil.compose)

    // Accompanist - permissions handling
    implementation(libs.accompanist.permissions)

    // Lifecycle
    implementation(libs.lifecycle.viewmodel.compose)
    implementation(libs.lifecycle.runtime.compose)

    // DataStore - local preferences
    implementation(libs.datastore.preferences)

    // Core Android
    implementation(libs.core.ktx)
    implementation(libs.activity.compose)

    // Vico Charts - telomere & genomic data visualization
    implementation(libs.vico.compose)
    implementation(libs.vico.compose.m3)
    implementation(libs.vico.core)

    // Testing
    testImplementation(libs.junit)
    androidTestImplementation(libs.espresso.core)
    androidTestImplementation(libs.compose.ui.test.junit4)
}
