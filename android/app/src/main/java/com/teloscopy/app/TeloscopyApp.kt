package com.teloscopy.app

import android.app.Application
import dagger.hilt.android.HiltAndroidApp

@HiltAndroidApp
class TeloscopyApp : Application() {

    override fun onCreate() {
        super.onCreate()
    }
}
