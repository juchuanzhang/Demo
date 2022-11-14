import easygui
a = easygui.ccbox(msg="老弟，还玩不？",title="询问",choices=["玩","不玩了"])
if a :
    easygui.msgbox("玩了好几把了，连个鸡屁股都没吃到，洗洗睡吧")
else:
    easygui.msgbox("ok,晚安老弟！！！")