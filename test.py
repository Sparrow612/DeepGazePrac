import base64
import os
import urllib.parse
import urllib.request
import urllib.error
import json
import ssl
import cv2
from main import HeadPoseEstimator

ssl._create_default_https_context = ssl._create_unverified_context

video_path = '/Users/chengrongxin/Downloads/gt2.1/videos'
videos = os.listdir(video_path)


def encode_img_base64(img_path) -> str:
    with open(img_path, 'rb') as f:
        img = f.read()
        img_in_base64 = base64.b64encode(img)
        return str(img_in_base64, 'utf-8')


def get_access_token(api_key="ESCHZFrtWtuqUbpbRCUNIE45", secret_key="NTwqnG6KYkycUq63ywiPuGCTFB3uLZl6"):
    host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=' + api_key + \
           '&client_secret=' + secret_key
    request = urllib.request.Request(host)
    request.add_header('Content-Type', 'application/json; charset=UTF-8')
    response = urllib.request.urlopen(request, timeout=100)
    content = response.read()
    response.close()
    if content:
        return json.loads(content)["access_token"]


access_token = get_access_token()


def baidu_body(image_base64):
    request_url = "https://aip.baidubce.com/rest/2.0/image-classify/v1/body_attr"
    access_token = get_access_token()
    try:
        data = {"image": image_base64}
        params = urllib.parse.urlencode(data).encode('utf-8')
        request_url = request_url + "?access_token=" + access_token
        request = urllib.request.Request(url=request_url, data=params)
        request.add_header('Content-Type', 'application/json')
        response = urllib.request.urlopen(request, timeout=100)
        content = response.read()
        response.close()
        if content:
            baidu_attributes = json.loads(str(content.decode('utf-8')))
            res = baidu_attributes
            return res
    except urllib.error.HTTPError as e:
        print("HTTPError!")
        print(e.code)
    except urllib.error.URLError as e:
        print("URLError!")
        print(e.reason)


def baidu_detection(image_base64):
    src_keys = ["expression", "glasses", "quality", "race", "angle"]
    dst_keys = ["smile", "glass", "facequality", "ethnicity", "headpose"]
    n = 5
    request_url = "https://aip.baidubce.com/rest/2.0/face/v3/detect"

    try:
        data = {"image": image_base64,
                "image_type": "BASE64",
                "face_field": "age,beauty,expression,face_shape,get_gender,glasses,landmark,"
                              "landmark72,landmark150,race,quality,eye_status,emotion,face_type"}
        params = urllib.parse.urlencode(data).encode('utf-8')
        request_url = request_url + "?access_token=" + access_token
        request = urllib.request.Request(url=request_url, data=params)
        request.add_header('Content-Type', 'application/json')
        response = urllib.request.urlopen(request, timeout=1000)
        content = response.read()
        response.close()
        if content:
            baidu_attributes = json.loads(str(content.decode('utf-8')))
            if baidu_attributes['error_code'] == 0:
                if baidu_attributes["result"]["face_list"][0]["race"]["type"] == "arabs":
                    baidu_attributes["result"]["face_list"][0]["race"][
                        "type"] = "yellow"
                for i in range(n):
                    baidu_attributes["result"]["face_list"][0][dst_keys[i]] = baidu_attributes["result"]["face_list"][
                        0].pop(src_keys[i])
            res = baidu_attributes
            try:
                return res['result']['face_list'][0]['headpose']
            except TypeError:
                return None

    except urllib.error.HTTPError as e:
        print("HTTPError!")
        print(e.code)
    except urllib.error.URLError as e:
        print("URLError!")
        print(e.reason)


out = 'frame.jpg'
result = {}
cur = 0
for video in videos:
    result['video' + str(cur)] = {}
    frames = result['video' + str(cur)]
    cur += 1
    v = os.path.join(video_path, video)
    cap = cv2.VideoCapture(v)
    if cap.isOpened():
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        for i in range(15, 25):
            cap.set(cv2.CAP_PROP_POS_FRAMES, (i * frame_count) // 26)
            # isRead 是否读取成功
            is_read, frame = cap.read()
            if frame is not None:
                frames[str(i - 15)] = {}

                cv2.imwrite('frame.jpg', frame)

                actual = HeadPoseEstimator(out).get_pitch_yaw_roll()
                for k in actual.keys():
                    actual[k] = str(actual[k])
                frames[str(i - 15)]['actual'] = actual

                expect = baidu_detection(encode_img_base64(out))
                if expect is not None:
                    for k in expect.keys():
                        expect[k] = str(expect[k])
                frames[str(i - 15)]['expected'] = expect

                print('DeepGaze:', actual)
                print('BaiduAPI:', expect)
f = open('test_result.json', 'w')
f.write(json.dumps(result))
f.close()
