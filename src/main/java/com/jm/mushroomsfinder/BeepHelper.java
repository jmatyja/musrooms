package com.jm.mushroomsfinder;

import android.media.AudioManager;
import android.media.ToneGenerator;
import android.os.Handler;
import android.os.Looper;

class BeepHelper
{
    private ToneGenerator toneG = new ToneGenerator(AudioManager.STREAM_ALARM, 100);

    public void beep(int duration)
    {
        toneG.startTone(ToneGenerator.TONE_DTMF_S, duration);
        Handler handler = new Handler(Looper.getMainLooper());
        handler.postDelayed(
                new Runnable() {
                    @Override
                    public void run() {
                        toneG.release();
                    }
                },
                (duration + 100));
    }
}