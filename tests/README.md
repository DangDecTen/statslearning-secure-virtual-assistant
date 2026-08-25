# Test The Core Logic


You can take a look at a succesful `test_core.py` run.

```text
Enrolled users: ['alice', 'bob']

--- OPEN command ---
  transcript      : 'cho tôi biết thời tiết hôm nay'
  intent          : get_weather (open)
  auth_passed     : True
  resolved_user_id: None
  response_text   : 'Hôm nay trời nắng nhẹ, nhiệt độ khoảng hai mươi tám độ.'
  audio_out       : data\tts_out\gtts_1786992324224.mp3

--- GATED command (claimed_user_id=alice) ---
  transcript      : 'Xem tin nhắn mới nhất của tôi'
  intent          : read_last_message (gated)
  auth_passed     : True
  resolved_user_id: alice
  response_text   : 'Đây là tin nhắn gần nhất của alice: (nội dung tin nhắn mẫu).'
  audio_out       : data\tts_out\gtts_1786992332893.mp3

  transcript      : 'Xem tin nhắn mới nhất của tôi'
  intent          : read_last_message (gated)
  auth_passed     : False
  resolved_user_id: None
  response_text   : 'Tôi không thể xác minh giọng nói của bạn.'
  audio_out       : data\tts_out\gtts_1786992341098.mp3

--- PERSONALIZED command ---
  transcript      : 'Hãy mở nhạc của tôi'
  intent          : play_my_music (personalized)
  auth_passed     : True
  resolved_user_id: alice
  response_text   : 'Đang phát danh sách nhạc yêu thích của alice.'
  audio_out       : data\tts_out\gtts_1786992349052.mp3

--- UNKNOWN command ---
  transcript      : 'Cả hai bên hãy cố gắng hiểu cho nhau.'
  intent          : unknown (none)
  auth_passed     : False
  resolved_user_id: None
  response_text   : 'Xin lỗi, tôi không hiểu yêu cầu của bạn.'
  audio_out       : None

All checks passed (see printed results above for auth correctness).
```
