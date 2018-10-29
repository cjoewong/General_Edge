# General Edge
This is a general framework for Edge Computing on Raspberry Pis. All modules are written in Python3.

### Table of Contents
- [Outline](#outline)
  * [Pi Manager](pi-manager)
  * [Config](config)
  * [Data Collector](data-collector)
  * [Algorithm](algorithm)
- [Get started](#get-started)
  * [Wifi and SSH Configuration](#wifi-and-ssh-configuration)
  * [Install](#install)
  * [Raspberry Pi Configuration](#raspberry-pi-configuration)
  * [Run](#run)
  
### Outline

#### Pi Manager
The <b>pi_manager.py</b> is the manager of the whole project, it will read the config file, learn the rules about this experiment, then use corresponding strategy to do data collection or run the algorithm.
#### Config
The config file contains the rules of the whole experiment, the general structure looks like:
```yaml
sensorPiA //name of the pi, you should specify this when you run the program.
    role:Algorithm/DataCollector //What should this Pi do.
    classPath: algorithm/data_collector //Path of the corresponding class you want to use.
    className: LinearRegression //The corresponding class.
    btAddress: XX:XX:XX:XX:XX:XX //Bluetooth address of this Pi.
    downStream: //The next handler.
    room: roomA //room in DynamoDB(optional)
    sensor: sensorA //sensor in DynamoDB(optional)
    dataFilePaths: //Path to your data(optional for data collector)
        - data/dataA
```
#### Data collector
The data collector will work like the sensor, it collects data and send it to its downstream. 
```py
def init() //Initialize this data collector, pass in necessary parameters.
def cleanup() //Clean this data collector.
def send() //Send data to the downstream.
def run() //Get data.
```

#### Algorithm
The algorithm module should contains all the logic you want your gateway to do. Generally, you should implement different ways to handle local computing and transfer data to downstream directly.
```py
def init() //Initialize this algorithm part, pass in necessary parameters.
def run() //Run the algorithm module.
def cleanup() //Clean this algorithm module.
def send() //Send data to the downstream.
```

### Get started

#### Wifi and SSH Configuration
If you want Raspberry Pis to connect Wifi and use SSH to login the Pis, you should put <b>utils/ssh</b> and <b>utils/wpa_supplicant.conf</b> in the root directory of your Raspberry Pi.

#### Configure AWS credential
```
export aws_access_key_id="XXXXXX"
export aws_secret_access_key="XXXXX"
```
#### Install
On every Raspberry Pi, clone this repository.
```sh
$ git clone https://github.com/YoungYang0820/General_Edge.git
```

#### Raspberry Pi Configuration
On every Raspberry Pi, you need to open bluetooth at first.
```sh
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
```sh
sudo pip3 install -r requirement.txt
```

#### Run
You should run based on your config file. 
```sh
$ python3 pi_manager.py base_cfg.yaml sensorPiA
$ python3 pi_manager.py base_cfg.yaml gatewayPiA
```

