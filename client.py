import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

s.connect((socket.gethostname(), 1234))
def cast(data):
   return str(data).rjust(1024, " ").encode("utf-8")

s.sendall(cast(input("Enter Number: ")))
s.sendall(cast(input("Enter Number: ")))
#s.sendall(sendable_data(input("Enter Number: ")))
s.sendall(cast(input("Enter Your Name: ")))

data = s.recv(1024)

print(f'Received:\n{data.decode("utf-8").strip()}') 