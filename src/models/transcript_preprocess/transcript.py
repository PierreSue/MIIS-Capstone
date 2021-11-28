def timestamp_to_seconds(time):
    time = time.split(':')
    return float(time[0])*3600 + float(time[1])*60 + float(time[2])


def transcript_preprocess(segments, caption_file):
    segment_boundaries_sec = [timestamp_to_seconds(t) for t in segments][::-1] # reverse the order to improve the time complexity
    with open(caption_file, 'r', encoding='utf8') as f:
        lines = f.readlines()

    captions, timesteps = '', []
    for i in range(0, len(lines), 3):
        end_time = timestamp_to_seconds(lines[i].split(',')[1])
        caption = lines[i+1].strip()

        if i == 0:
            captions = caption
        else:
            captions = ' '.join([captions, caption])

        timesteps.append((end_time, len(captions)))

    i, start_time = 0, 0
    transcript_segments = []
    while i < len(timesteps) and len(segment_boundaries_sec) > 0:
        timestep, cutpoint = timesteps[i]
        if segment_boundaries_sec[-1] < timestep:
            transcript_segments.append(captions[start_time:cutpoint].replace('um ', ''))
            start_time = cutpoint
            segment_boundaries_sec.pop(-1)
        else:
            i += 1
    
    if len(segment_boundaries_sec) != 0:
        transcript_segments += ['']*(len(segments) + 1 - len(transcript_segments))
    else:
        transcript_segments.append(captions[start_time:])

    return transcript_segments

if __name__ == '__main__':
    bounaries = ['00:00:22', '00:00:54', '00:01:44', '00:03:34', '00:04:34', '00:17:20', '00:18:30', '00:40:14', '00:45:52', '01:05:44', '01:06:50', '01:11:08', '01:11:46', '01:16:36', '01:17:58']
    caption_file = '../../../../data/youtube_asr/11645_1.sbv'
    segment_transcript = transcript_preprocess(bounaries, caption_file)