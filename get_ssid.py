import os

#'\' is used to splite pythone line
ipaddress = os.popen("ifconfig wlan0| grep 'inet addr'| awk -F: '{print $2}' | awk '{print $1}'").read()
ssid = os.popen("iwconfig wlan0 | grep 'ESSID' | awk '{print $4}' | awk -F\\\" '{print $2}'").read()

print("ssid: " + ssid)
print("ipaddress: " + ipaddress)
