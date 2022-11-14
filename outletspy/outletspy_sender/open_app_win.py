#!/usr/bin/python
# -*- coding: utf-8 -*-

import win32api
import time
import os


"""
# the absolute path of exe file
'C:/Program Files (x86)/Microsoft Office/root/Office16/WINWORD.EXE',
'C:/Program Files (x86)/Microsoft Office/root/Office16/POWERPNT.EXE',
'C:/Program Files (x86)/Microsoft Office/root/Office16/EXCEL.EXE',
'D:/Program Files (x86)/MATLAB/R2016b/bin/matlab.exe',
'D:/Program Files (x86)/Foxit Software/Foxit Reader/FoxitReader.exe',
'C:/Program Files/Internet Explorer/iexplore.exe'
'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
'D:/Program Files (x86)/Tencent/QQLive/9.21.2159.0/QQLive.exe',
'D:/Program Files (x86)/KMPlayer/KMPlayer.exe',
'D:/Program Files (x86)/Baofeng/StormPlayer/StormPlayer.exe',
'D:/Program Files (x86)/Netease/CloudMusic/cloudmusic.exe',
"""
a=[
   r"C:\Users\usslab\AppData\Local\Wunderlist\Wunderlist.exe",
   r"C:\Program Files\WinRAR\WinRAR.exe",
   r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
   r"C:\Program Files\CCleaner\CCleaner.exe",
   r"C:\Program Files\VideoLAN\VLC\vlc.exe",
   r"C:\Users\usslab\AppData\Local\keeperpasswordmanager\keeperpasswordmanager.exe",
   r"C:\Users\usslab\AppData\Local\WhatsApp\WhatsApp.exe",
   r"C:\Program Files (x86)\Steam\Steam.exe",
   r"C:\Program Files (x86)\Microsoft\Skype for Desktop\Skype.exe",
   r"C:\Program Files (x86)\iTunes\iTunes.exe",
   r"C:\Program Files (x86)\Notepad++\notepad++.exe",
   r"C:\Program Files (x86)\SumatraPDF\SumatraPDF.exe",
   r"C:\Program Files (x86)\Dropbox\Client\Dropbox.exe" ,
   r"C:\Program Files\Microsoft Office\Office16\POWERPNT.exe",
   r"C:\Program Files\Microsoft Office\Office16\WINWORD.exe",
   r"C:\Program Files\Microsoft Office\Office16\EXCEL.exe"
   ]
b=["Wunderlist.exe","WinRAR.exe","chrome.exe","CCleaner64.exe","vlc.exe",
   "keeperpasswordmanager.exe","WhatsApp.exe","Steam.exe","Skype.exe","iTunes.exe",
   "notepad++.exe","SumatraPDF.exe","Dropbox.exe","POWERPNT.exe","WINWORD.exe","EXCEL.exe"]
##############################################################
## static variables
#program = "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
#programName = "chrome.exe"

staticItvl = 5
openItvl = 7
closeItvl = 5
repeatCnt = 1
outputDir = "C:/timestamp/"
##############################################################
background = [
    r"C:\Program Files (x86)\Windows Media Player\wmplayer.exe",
    r"C:\Program Files\internet explorer\iexplore.exe",

    r"C:\Users\usslab\AppData\Local\Programs\Microsoft VS Code\Code.exe",
    r"C:\Windows\system32\mstsc.exe"]
background_close = ["wmplayer.exe","iexplore.exe","Code.exe","mstsc.exe"]
if __name__ == '__main__':
    for k in range(1):
        # win32api.ShellExecute(0, 'open', background[k], '','',1)
        for i in range(16):
            program=a[i]
            programName=b[i] 
            ## get filename
            tm = time.gmtime()
            filename = outputDir + time.strftime("%Y%m%d%H%M%S", tm) + "_" + programName + "_time.txt"
            print("Output File: " + filename)


            with open(filename, 'w') as f_output:
                ## black period as the baseline
                print("Blank Period: " + str(staticItvl) + "s.")
                time.sleep(staticItvl)

                ## start to repeatly open and close the app
                for i in range(repeatCnt):
                    printline = "Open #" + str(i+1) + "/" + str(repeatCnt) + " " + programName
                    print(printline)

                    ## open an app
                    c_time = str(time.time())
                    logstamp = "%s open\n" % (c_time)
                    f_output.write(logstamp)

                    win32api.ShellExecute(0, 'open', program, '','',1)
                    #win32api.ShellExecute(0, 'open', program, '','',1)
                    #win32api.ShellExecute(0, 'open', program, '','',1)
                    #win32api.ShellExecute(0, 'open', program, '','',1)
                    #win32api.ShellExecute(0, 'open', program, '','',1)
                    #win32api.ShellExecute(0, 'open', program, '','',1)
                    #win32api.ShellExecute(0, 'open', program, '','',1)
                    #win32api.ShellExecute(0, 'open', program, '','',1)
                    #win32api.ShellExecute(0, 'open', program, '','',1)
                    ## sleep before close the app
                    time.sleep(closeItvl)


        
                    # close the app
                    c_time = str(time.time())
                    logstamp = "%s close\n" % (c_time)
                    f_output.write(logstamp)

                    os.system("taskkill /F /IM "+ programName)
                    #os.system("taskkill /F /IM "+ programName)
                    #os.system("taskkill /F /IM "+ programName)
                    #os.system("taskkill /F /IM "+ programName)
                    #os.system("taskkill /F /IM "+ programName)
                    #os.system("taskkill /F /IM "+ programName)
                    #os.system("taskkill /F /IM "+ programName)
                    #os.system("taskkill /F /IM "+ programName)
                    #os.system("taskkill /F /IM "+ programName)
                    
                    ## sleep before open the app again
                    time.sleep(openItvl-closeItvl)

            f_output.close()
        os.system("taskkill /F /IM "+ background_close[k])
        