from flask import Flask, request, jsonify, send_file, render_template
import threading
import cv2
from ultralytics import YOLO
import numpy as np
import time
from PIL import ImageFont, ImageDraw, Image
import math
from io import BytesIO
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
import matplotlib
import matplotlib.dates as mdates
import io
import sounddevice as sd
from scipy.io import wavfile
from matplotlib.font_manager import FontProperties
import pickle

app = Flask(__name__)
matplotlib.use('Agg')

#查找摄像头
deviceidx=0
for device_id in range(15):
    cap = cv2.VideoCapture(device_id)
    if not cap.isOpened():
        cap.release()
        continue
    ret, frame = cap.read()
    if ret and frame is not None:
        deviceidx = device_id
        print(str(device_id)+' is available')
    cap.release()
print(deviceidx)
# 初始化全局变量，用于存储状态和数据

cap = None
model = None
hist = []
chist = []
af,interval,cstate=None,10,0
font = ImageFont.truetype('sarasa-mono-sc-regular.ttf', size=21)
font_path = 'sarasa-mono-sc-regular.ttf'
font_prop = FontProperties(fname=font_path)
calibflag=0

stdval=[27, 113, 70, 1, 103, 79, 4]
tol=[[-0.1,0.2],[-4.5,6.5],[-0.1,0.2],[-4.5,6.5],[-13,13],[-13,13],[-25,25]]
istbc=0
lstudyt=45
islbc=0
alarms=[]
longhistory=[]

#load from pickle
with open('var/stdval.pkl', 'rb') as file:
    stdval = pickle.load(file)
with open('var/tol.pkl', 'rb') as file:
    tol = pickle.load(file)
with open('var/istbc.pkl', 'rb') as file:
    istbc = pickle.load(file)
with open('var/lstudyt.pkl', 'rb') as file:
    lstudyt = pickle.load(file)
with open('var/islbc.pkl', 'rb') as file:
    islbc = pickle.load(file)
with open('var/alarms.pkl', 'rb') as file:
    alarms = pickle.load(file)
with open('var/longhistory.pkl', 'rb') as file:
    longhistory = pickle.load(file)
tol=[[-0.1,0.2],[-4.5,6.5],[-0.1,0.2],[-4.5,6.5],[-13,13],[-13,13],[-25,25]]

#hs
def calib(kval,toll):
    global stdval,tol
    stdval=kval
    tol=toll
    print(stdval,tol)
    return stdval,tol
    
def playaudio(idx):
    file_path = 'audio/'+str(idx)+'.wav'
    rate, data = wavfile.read(file_path)
    sd.play(data, rate)
    #sd.wait()
    
def get_greeting():
    global hist,cstate,chist
    time_stat=[len(hist),0,0]#开机 学习 坐姿错误
    for i in range(len(hist)):
        if hist[i][0]==0:
            time_stat[1]=time_stat[1]+1
        elif hist[i][0]>0:
            time_stat[1]=time_stat[1]+1
            time_stat[2]=time_stat[2]+1
    return '今日开机'+str(int(time_stat[0]/6))+'分钟,学习'+str(int(time_stat[1]/6))+'分钟,其中坐姿不正确'+str(int(time_stat[2]/6))+'分钟'

def get_greetchart():
    global hist,chist
    status_labels = ['没人', '正常', '低头', '歪头', '趴桌', '脖子前伸', '身体下沉', '驼背写字']
    status = [item[0]+1 for item in hist]
    times = [item[1]-hist[0][1] for item in hist]
    plt.figure(figsize=(10, 6))
    plt.plot(times, status, marker='o', linestyle='-', color='b')
    plt.yticks(range(len(status_labels)), status_labels,fontproperties=font_prop)
    plt.gcf().autofmt_xdate()
    img = io.BytesIO()
    plt.savefig(img, format='jpg')
    img.seek(0)
    plt.close()
    return send_file(img, mimetype='image/jpeg')

def get_af():
    global af
    _, buffer = cv2.imencode('.jpeg', af)
    io_buf = BytesIO(buffer)
    return send_file(io_buf, mimetype='image/jpeg')
    
def get_recentstat():
    global hist, chist
    statt=[]
    typ,st,ed,cur=0,0,0,0
    for i in range(len(chist)-1):
        if chist[i]!=0 and cur==0:
            st=i
            typ=hist[i]
            cur=1
        if chist[i]==0 and cur!=0:
            ed=i
            cur=0
            statt.append([typ[0],hist[st][1],hist[ed][1]])
    if cur!=0:
        ed=len(chist)-1
        statt.append([typ[0],hist[st][1],hist[ed][1]])
    print(statt)
    return statt
    
def set_studytime(tt):
    global lstudyt
    lstudyt=tt
    
def set_alarm(ala):
    global alarms
    alarms.append(ala)
    with open('var/alarms.pkl', 'wb') as file:
        pickle.dump(alarms, file)
        
def get_alarm():
    global alarms
    return alarms

def delalm(idx):
    global alarms
    rett=alarms.pop(idx)
    with open('var/alarms.pkl', 'wb') as file:
        pickle.dump(alarms, file)
    return rett

def time_broadcast():
    global istbc
    last_broadcast_time = None
    while True:
        current_time = datetime.now()
        current_hour = current_time.hour
        current_minute = current_time.minute
        current_second = current_time.second
        if istbc == 1 and 8 <= current_hour <= 22:
            if current_minute == 0 and current_second <= 15:
                if last_broadcast_time != current_hour:
                    playaudio(current_hour+100)
                    last_broadcast_time = current_hour
                    time.sleep(100)
        time.sleep(1)
broadcast_thread = threading.Thread(target=time_broadcast)
broadcast_thread.daemon = True
broadcast_thread.start()

def time_alarm():
    global alarms
    while True:
        for i in range(len(alarms)):
            if alarms[i][0]<time.time():
                playaudio(205)
                if alarms[i][1]>0:
                    alarms[i][0]+=alarms[i][1]
                    with open('var/alarms.pkl', 'wb') as file:
                        pickle.dump(alarms, file)
                else:
                    delalm(i)
        time.sleep(1)
alarm_thread = threading.Thread(target=time_alarm)
alarm_thread.daemon = True
alarm_thread.start()

def lsl_broadcast():
    global islbc, hist, lstudyt
    while True:
        if islbc == 1:
            if len(hist) > lstudyt * 6:
                last_lstudyt = [x[0] for x in hist[-lstudyt*6:]]
                slen=0
                for i in range(len(last_lstudyt)):
                    if last_lstudyt[i]>0:
                        slen=slen+1
                if slen / len(last_lstudyt) > 0.96:
                    playaudio(201)
                    time.sleep(300)
        time.sleep(1)
ls_thread = threading.Thread(target=lsl_broadcast)
ls_thread.daemon = True
ls_thread.start()

def set_timebroadcast(setq):
    global istbc
    istbc=setq
    
def get_historystat():
    global hist, chist
    statt=[]
    typ,st,ed,cur=0,0,0,0
    for i in range(len(chist)-1):
        if chist[i]!=0 and cur==0:
            st=i
            typ=hist[i]
            cur=1
        if chist[i]==0 and cur!=0:
            ed=i
            cur=0
            statt.append([typ[0],hist[st][1],hist[ed][1]])
    if cur!=0:
        ed=len(chist)-1
        statt.append([typ[0],hist[st][1],hist[ed][1]])
    print(statt)
    return statt
    
def get_fhistorystat():
    global hist, chist, longhistory
    statt = []
    typ, st, ed, cur = 0, 0, 0, 0
    for i in range(len(chist) - 1):
        if chist[i] != 0 and cur == 0:
            st = i
            typ = hist[i]
            cur = 1
        if chist[i] == 0 and cur != 0:
            ed = i
            cur = 0
            statt.append([typ[0],hist[st][1],hist[ed][1]])
    if cur != 0:
        ed = len(chist) - 1
        statt.append([typ[0],hist[st][1],hist[ed][1]])
    statt = longhistory + statt
    return statt

def get_webmain():
    return render_template('main.html')
    

# 将处理视频的逻辑放在一个单独的线程中执行
def video_processing_thread():
    global cap, model, hist, chist, stdval, tol,font,af,interval,cstate,calibflag,longhistory
    def are_same_day(timestamp1, timestamp2):
        dt1 = datetime.fromtimestamp(timestamp1).astimezone()
        dt2 = datetime.fromtimestamp(timestamp2).astimezone()
        return dt1.date() == dt2.date()
    
    def draw_CHtext_on_image(image, text, position, color):
        global font
        pil_image = Image.fromarray(image.copy())
        draw = ImageDraw.Draw(pil_image)
        text_x, text_y = position
        draw.text((text_x, text_y), text, font=font, fill=color)
        return np.array(pil_image)
    
    def calkv(kpts):
        head_shoulder_dist = np.linalg.norm(kpts[0] - kpts[1])
        head_shoulder_vertical_angle = math.degrees(math.atan2(kpts[1,1] - kpts[0,1], kpts[1,0] - kpts[0,0]))
        waist_shoulder_dist = np.linalg.norm(kpts[4] - kpts[1])
        waist_shoulder_vertical_angle = math.degrees(math.atan2(kpts[1,1] - kpts[4,1], kpts[1,0] - kpts[4,0]))+90
        elbow_shoulder_horizontal_angle = math.degrees(math.atan2(kpts[2,1] - kpts[1,1], kpts[2,0] - kpts[1,0]))
        hand_shoulder_horizontal_angle = math.degrees(math.atan2(kpts[3,1] - kpts[1,1], kpts[3,0] - kpts[1,0]))
        vision_angle=180-math.degrees(math.atan2(kpts[0,1] - kpts[5,1], kpts[0,0] - kpts[5,0]))
        kval = [head_shoulder_dist,head_shoulder_vertical_angle,waist_shoulder_dist,waist_shoulder_vertical_angle,elbow_shoulder_horizontal_angle,hand_shoulder_horizontal_angle,vision_angle]
        print(kval)#头与肩膀距离长度,头-肩膀-竖直线构成的夹角(deg),腰与肩膀距离长度,腰-肩膀-竖直线构成的夹角(deg),手肘-肩膀-水平线构成的夹角(deg),头-肩膀-水平线构成的夹角(deg),视角
        return kval
    
    def annotate(frame,kval,actl,kpts):
        global stdval,tol
        res,color,typ = [],[True,True,True,True,True,True,True,True,True,True,True],0
        for i in range(len(kval)):#cmp
            diff = kval[i] - stdval[i]
            if diff >= tol[i][0] and diff <= tol[i][1]:
                res.append(True)
            else:
                res.append(False)
        lk1=(kval[0] - stdval[0])/stdval[0]
        lk2=(kval[2] - stdval[2])/stdval[2]
        if lk1>= tol[0][0] and lk1 <= tol[0][1]:
            res[0]=True
        else:
            res[0]=False
        if lk2>= tol[2][0] and lk2 <= tol[2][1]:
            res[2]=True
        else:
            res[2]=False
        if not res[0]:
            color[3]=False
            typ=2
        if not res[1]:
            color[2],color[4]=False,False
            typ=4
        if not res[2]:
            color[5]=False
            typ=6
        if not res[3]:
            color[4],color[6]=False,False
            typ=5
        if not res[4]:
            color[4],color[7],color[8]=False,False,False
            typ=3
        if not res[5]:
            color[8],color[9],color[10]=False,False,False
            typ=3
        if not res[6]:
            color[0],color[1],color[2]=False,False,False
            typ=1
        kptdraw=np.array([kpts[5],kpts[0],kpts[1],kpts[4],kpts[2],kpts[3]])
        for i in range(len(kptdraw)):
            cv2.circle(img=frame,center=(int(kptdraw[i][0]), int(kptdraw[i][1])),radius=5,color=(0, 255, 0) if color[i*2] else (0, 0, 255),thickness=-1)
            if (i+1)<4:
                cv2.line(img=frame,pt1=(int(kptdraw[i][0]), int(kptdraw[i][1])),pt2=(int(kptdraw[i+1][0]), int(kptdraw[i+1][1])),color=(255, 0, 0) if color[i*2+1] else (0, 0, 255),thickness=2)
        cv2.line(img=frame,pt1=(int(kptdraw[2][0]), int(kptdraw[2][1])),pt2=(int(kptdraw[4][0]), int(kptdraw[4][1])),color=(255, 0, 0) if color[7] else (0, 0, 255),thickness=2)
        cv2.line(img=frame,pt1=(int(kptdraw[4][0]), int(kptdraw[4][1])),pt2=(int(kptdraw[5][0]), int(kptdraw[5][1])),color=(255, 0, 0) if color[7] else (0, 0, 255),thickness=2)
        frame=draw_CHtext_on_image(frame, actl[typ], (2,2), 'green' if typ==0 else 'blue')
        return frame,typ
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        start_t=time.time()
        iperson=-1
        if calibflag==1:
            cap.read()
            cap.read()
            cap.read()
        ret, frame = cap.read()
        ret, frame = cap.read()
        ret, frame = cap.read()
        if not ret:
            print("Cannot read video stream")
            break
        frame=cv2.rotate(frame,cv2.ROTATE_90_COUNTERCLOCKWISE)
        results = model(frame)
        if len(results[0].boxes.cls.numpy())>0:
            iperson=1
            rs=results[0].keypoints.data.cpu().numpy()[0]
            lhs,rhs=0,0
            for i in range(0,6):
                lhs=lhs+(rs[2*i+1][2])*(rs[2*i+1][2])
                rhs=rhs+(rs[2*i+2][2])*(rs[2*i+2][2])
                #(lhs,rhs)
            #print(rs)
            rs=results[0].keypoints.xy.cpu().numpy()[0]
            if lhs>rhs:
                kpts=np.array([rs[3],rs[5],rs[7],rs[9],rs[11],rs[1]])
            else:
                kpts=np.array([rs[4],rs[6],rs[8],rs[10],rs[12],rs[2]])
            #print(kpts)#头 肩膀 手肘 手 腰 眼睛
            kval=calkv(kpts)#
            actl=['正常','低头','歪头','趴桌','脖子前伸','身体下沉','驼背写字']
            af,rs=annotate(frame,kval,actl,kpts)
            if calibflag==1:
                calib(kval,tol)
                calibflag=0
                playaudio(202)
        else:
            af=frame
            print('No person detected')
        if len(hist)>10:
            if not are_same_day(hist[len(hist)-1][1],time.time()):
                longhistory=get_historystat()+longhistory
                with open('var/longhistory.pkl', 'wb') as file:
                    pickle.dump(longhistory, file)
                hist=[]
                chist=[]
                
        hist.append([rs if iperson==1 else -1,time.time()])
        #annotated_frame = results[0].plot()
        if len(hist)>3:
            if hist[-1][0]>0 and hist[-2][0]>0 and hist[-3][0]>0 and cstate==0:
                cstate=1
                playaudio(hist[-1][0])
            if hist[-1][0]==0 and hist[-2][0]==0 and cstate!=0:
                cstate=0
        chist.append(cstate)
        #cv2.imshow('Pose Detection', af)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        print(time.time()-start_t)
        if time.time()-start_t<interval:
            time.sleep(interval-(time.time()-start_t))
        #get_recentstat()
    cv2.waitKey(0)
    cap.release()

@app.route('/jlc', methods=['GET'])
def istest():
    return 'jlcjlcjlc'
@app.route('/api/get_greeting', methods=['GET'])
def api_get_greeting():
    return get_greeting()
@app.route('/api/get_af', methods=['GET'])
def api_get_af():
    return get_af()
@app.route('/api/get_recentstat', methods=['GET'])
def api_get_recentstat():
    return get_recentstat()
@app.route('/api/get_greetchart', methods=['GET'])
def api_get_greetchart():
    return get_greetchart()
@app.route('/main', methods=['GET'])
def mainpage():
    return get_webmain()

@app.route('/api/get_alarm', methods=['GET'])
def api_get_alarm():
    return jsonify(get_alarm())
@app.route('/api/delete_alarm/<int:idx>', methods=['DELETE'])
def api_delete_alarm(idx):
    delalm(idx)
    return jsonify({"message": "Alarm deleted successfully"})
@app.route('/api/set_alarm', methods=['POST'])
def api_set_alarm():
    data = request.get_json()
    if data:
        alarm_time = data[0]
        repeat_interval = data[1]
        alarm = [alarm_time, repeat_interval]
        print(alarm)
        set_alarm(alarm)
        return jsonify({"message": "Alarm set successfully"})
    else:
        return jsonify({"message": "Invalid alarm data"}), 400
@app.route('/new_alarm', methods=['GET'])
def newalarm():
    return render_template('new_alarm.html')

@app.route('/view_history', methods=['GET'])
def view_history():
    return render_template('view_history.html')
@app.route('/api/get_historystat', methods=['GET'])
def api_get_historystat():
    statt = get_fhistorystat()
    return jsonify(statt)

@app.route('/settings', methods=['GET'])
def settings():
    return render_template('settings.html')
@app.route('/api/istbc', methods=['GET'])
def api_istbc():
    global istbc
    return jsonify({"status": istbc})
@app.route('/api/change_istbc', methods=['POST'])
def api_change_istbc():
    global istbc
    istbc = 1 - istbc
    with open('var/istbc.pkl', 'wb') as file:
        pickle.dump(istbc, file)
    return jsonify({"status": istbc})
@app.route('/api/islbc', methods=['GET'])
def api_islbc():
    global islbc
    return jsonify({"status": islbc})
@app.route('/api/change_islbc', methods=['POST'])
def api_change_islbc():
    global islbc
    islbc = 1 - islbc
    with open('var/islbc.pkl', 'wb') as file:
        pickle.dump(islbc, file)
    return jsonify({"status": islbc})
@app.route('/api/set_ls', methods=['POST'])
def api_set_ls():
    global lstudyt
    data = request.get_json()
    if data:
        lstudyt = data['value']
        with open('var/lstudyt.pkl', 'wb') as file:
            pickle.dump(lstudyt, file)
        return jsonify({"message": "Success"})
    else:
        return jsonify({"message": "Invalid data"}), 400

@app.route('/calib', methods=['GET'])
def calibb():
    return render_template('calib.html')
@app.route('/api/calib', methods=['POST'])
def api_calib():
    global stdval, tol, calibflag
    calibflag=1
    return jsonify({"message": "Success"})

if __name__ == '__main__':
    model = YOLO('pose.pt')
    cap = cv2.VideoCapture(deviceidx)
    #cap = cv2.VideoCapture("testvideo.mp4")
    cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
    thread = threading.Thread(target=video_processing_thread)
    thread.daemon = True
    thread.start()
    app.run(port=81, host='10.42.0.1', use_reloader=False)
    #app.run(use_reloader=False)