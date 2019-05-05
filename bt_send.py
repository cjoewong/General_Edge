from utils import bluetootch_utils

data = 'hello niggaz \0'
down_addr = 'B8:27:EB:68:A5:4C'

bluetootch_utils.sendData(data, down_addr, 1)