# ============================================================================
# ProGuard Rules for Teloscopy
# Genomic Intelligence Platform
# ============================================================================

# ---- Retrofit ----
# Keep Retrofit interfaces and their annotations
-keepattributes Signature
-keepattributes Exceptions
-keepattributes *Annotation*

-keepclasseswithmembers class * {
    @retrofit2.http.* <methods>;
}

-keep,allowobfuscation interface * {
    @retrofit2.http.* <methods>;
}

-dontwarn retrofit2.**
-keep class retrofit2.** { *; }

# ---- OkHttp ----
-dontwarn okhttp3.**
-dontwarn okio.**
-keep class okhttp3.** { *; }
-keep interface okhttp3.** { *; }

# ---- Moshi ----
# Keep Moshi adapters and JSON model classes
-keep class com.squareup.moshi.** { *; }
-keep interface com.squareup.moshi.** { *; }

-keepclassmembers class * {
    @com.squareup.moshi.Json <fields>;
}

-keepclassmembers class * {
    @com.squareup.moshi.FromJson <methods>;
    @com.squareup.moshi.ToJson <methods>;
}

# Keep generated Moshi adapters
-keep class **JsonAdapter {
    <init>(...);
    <fields>;
}

-keepnames @com.squareup.moshi.JsonClass class *

# ---- Hilt / Dagger ----
-keep class dagger.hilt.** { *; }
-keep class javax.inject.** { *; }
-keep class * extends dagger.hilt.android.internal.managers.ViewComponentManager$FragmentContextWrapper { *; }

-keepclassmembers class * {
    @dagger.* <fields>;
    @dagger.* <methods>;
    @javax.inject.* <fields>;
    @javax.inject.* <methods>;
}

# Keep Hilt generated components
-keep class *_HiltModules* { *; }
-keep class *_HiltComponents* { *; }
-keep class *_GeneratedInjector { *; }
-keep class * implements dagger.hilt.internal.GeneratedComponent { *; }

# ---- Teloscopy API Models ----
# Keep all data classes in the API package for serialization
-keep class com.teloscopy.app.data.api.** { *; }
-keepclassmembers class com.teloscopy.app.data.api.** {
    <init>(...);
    <fields>;
}

# ---- Kotlin ----
-keep class kotlin.Metadata { *; }
-dontwarn kotlin.**

# Keep Kotlin coroutines
-keepnames class kotlinx.coroutines.internal.MainDispatcherFactory {}
-keepnames class kotlinx.coroutines.CoroutineExceptionHandler {}
-keepclassmembers class kotlinx.coroutines.** {
    volatile <fields>;
}

# ---- Compose ----
-dontwarn androidx.compose.**

# ---- CameraX ----
-keep class androidx.camera.** { *; }

# ---- Vico Charts ----
-keep class com.patrykandpatrick.vico.** { *; }

# ---- General ----
# Keep source file names and line numbers for crash reports
-keepattributes SourceFile,LineNumberTable
-renamesourcefileattribute SourceFile

# Remove logging in release builds
-assumenosideeffects class android.util.Log {
    public static boolean isLoggable(java.lang.String, int);
    public static int v(...);
    public static int d(...);
    public static int i(...);
}
