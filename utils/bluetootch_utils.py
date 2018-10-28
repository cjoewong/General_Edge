import bluetooth
import time
import pickle
import logging

def listenOnBluetooth(channel):
    '''
    Listen to incoming data on Bluetooth Interface
    '''
    logger = logging.getLogger()
    allowableUnacceptedConns = 1 # The # of unaccepted connection before refusing new connections
    bufferSize = 1 # Receive up to this number of buffersize bytes from the socket

    # Setup the Bluetooth connection
    server_sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    server_sock.bind(("", channel))
    server_sock.listen(allowableUnacceptedConns)

    # startTime = time.time()
    total_data = []

    # Listen for incoming data, while watching for the keyboard interrupt
    try:
        # The received address is a (host, channel) tuple
        client_sock, address = server_sock.accept()
        start_time = time.time()

        while True:
            # startTime = time.time()
            data_1 = client_sock.recv(bufferSize)
            # We need a break statement because when no data is available,
            # recv() blocks until at least one byte is available
            if len(data_1) == 0:
                break
            # Append the received data to the helper variable
            total_data.append(data_1)

        # Should we stop timing at this point?
        # endTime = time.time()

    except IOError:
        pass    # Sincere apologies to all who told me passing is poor practice

    except KeyboardInterrupt:
        bluetooth.stop_advertising(server_sock)
        # sys.exit()    Why do we need an exit before we're actually done?

    except Exception:
        print('No data is received, waiting...')
        time.sleep(5)

    # Log the results of the bluetooth data to the console
    end_time = time.time()
    bt_time = end_time - start_time

    logger.info("Bluetooth Transmission Time %f", bt_time)

    # Close the bluetooth connection
    client_sock.close()
    server_sock.close()

    return bt_time, pickle.loads(b''.join(total_data))


def sendData(data, target_address, channel):
    '''
    Send data over Bluetooth.
    '''
    logger = logging.getLogger()
    start_time = time.time()

    # Establish BT connection
    while True:
        try: 
            sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
            sock.connect((target_address, channel))

            # Send data
            data_to_send = pickle.dumps(data)
            data_size = len(data_to_send)
            bytes_sent = sock.send(data_to_send)

            sock.close()
            break
        except Exception as e:
            print('Dest is not available, try in 5s... {0}'.format(str(e)))
            time.sleep(5)    

    end_time = time.time()
    bt_time = end_time - start_time
    logger.info("Sensor : Sent %f/%f bytes over Bluetooth in %f seconds", bytes_sent, data_size, bt_time)
