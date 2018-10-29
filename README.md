# General Edge
This is a general framework for Edge Computing on Raspberry Pis. All modules are written in Python3.

# Table of Contents
- [Get started](#get-started)
  * [Wifi and SSH Configuration](#wifi-and-ssh-configuration)
  * [Install](#install)
  * [Raspberry Pi Configuration](#raspberry-pi-configuration)
  * [Run](#run)
- [Outline](#outline)
  * [pi_manager.py](pi-manager)
  * [config.yaml](config)
  * [data collector](data-collector)
  * [algorithm](algorithm)
  
# Get started

## Wifi and SSH Configuration
If you want Raspberry Pis to connect Wifi and use SSH to login the Pis, you should put <b>utils/ssh</b> and <b>utils/wpa_supplicant.conf</b> in the root directory of your Raspberry Pi.

## Install
On every Raspberry Pi, clone this repository.
```
$ git clone https://github.com/YoungYang0820/General_Edge.git
```

## Raspberry Pi Configuration
On every Raspberry Pi, you need to open bluetooth at first.
```
$ sudo apt-get install bluetooth libbluetooth-dev
$ sudo python3 -m pip install pybluez

$ sudo /etc/init.d/bluetooth restart
$ sudo bluetoothctl
    $ scan on
    $ power on
    $ agent on
    $ discoverable on
```

Then, install the corresponding packages.
```
sudo pip3 install -r requirement.txt
```

## Run
You should run based on your config file. 
```
$ python3 pi_manager.py base_cfg.yaml sensorPiA
$ python3 pi_manager.py base_cfg.yaml gatewayPiA
```
