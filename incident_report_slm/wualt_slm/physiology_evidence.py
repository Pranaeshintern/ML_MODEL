"""Deterministic measurement facts. No diagnosis, emotion or danger scoring."""


def physiology_facts(data):
    p = data.physiology
    facts = []
    h = p.hrv
    if h:
        seconds = (h.window_end-h.window_start).total_seconds()
        current = h.window_start <= data.observation_window.end and h.window_end >= data.observation_window.start
        reasons = []
        r = h.reference
        if not current:
            reasons.append('measurement window does not overlap the incident observation window')
        if h.signal_quality != 'usable':
            reasons.append('signal quality is ' + h.signal_quality)
        if h.artifact_corrected is not True:
            reasons.append('artifact correction is not confirmed')
        if r is None:
            reasons.append('no personal reference supplied')
        else:
            if h.metric != r.metric:
                reasons.append('reference uses a different HRV metric')
            if seconds != r.window_seconds:
                reasons.append('reference uses a different recording duration')
            if r.comparable_context is not True:
                reasons.append('comparable recording conditions are not confirmed')
        comparison = 'unavailable'
        if not reasons:
            comparison = 'above' if h.value_ms > r.value_ms else 'below' if h.value_ms < r.value_ms else 'equal'
        text = f'HRV {h.metric.upper()}: {h.value_ms:g} ms over {seconds:g} seconds; signal quality {h.signal_quality}.'
        text += ' Artifact correction ' + ('confirmed.' if h.artifact_corrected is True else 'not confirmed.')
        if r:
            text += f' Supplied reference: {r.metric.upper()} {r.value_ms:g} ms over {r.window_seconds:g} seconds.'
        text += (' Comparison not supported: ' + '; '.join(reasons) + '.' if reasons
                 else f'Current {h.metric.upper()} is {comparison} the comparable supplied reference; this does not establish a cause or emotional state.')
        facts.append({'id':'V', 'kind':'measurement', 'text':text, 'available':True,
                      'record_text':text + f' Window: {h.window_start.isoformat()} to {h.window_end.isoformat()}.'
                                    + (f' Reference recorded: {r.at.isoformat()}.' if r else ''),
                      'metric':h.metric, 'value_ms':h.value_ms, 'window_seconds':seconds,
                      'signal_quality':h.signal_quality, 'current_window_overlap':current,
                      'comparison':comparison, 'comparison_limitations':reasons,
                      'at':h.window_end.isoformat()})
    labels = {'spo2_percent': ('SpO2', '%'),
              'respiratory_rate_per_minute': ('Respiratory rate', 'breaths/minute'),
              'skin_temperature_c': ('Skin temperature', 'degrees C')}
    for i, m in enumerate(p.other_measurements, 1):
        label, unit = labels[m.name]
        current = data.observation_window.start <= m.at <= data.observation_window.end
        text = f'{label}: {m.value:g} {unit}; signal quality {m.signal_quality}.'
        if not current:
            text += ' This reading is outside the incident observation window.'
        if m.name == 'skin_temperature_c':
            text += ' Skin temperature is not core body temperature.'
        facts.append({'id':f'P{i:02}', 'kind':'measurement', 'text':text,
                      'record_text':text + ' Recorded: ' + m.at.isoformat() + '.',
                      'available':True, 'signal_quality':m.signal_quality,
                      'current_window_overlap':current, 'at':m.at.isoformat()})
    return facts
