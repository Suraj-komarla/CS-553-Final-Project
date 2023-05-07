Create two EC2 instances with the following security rules enabled:
	1) Allow all TCP traffic
	2) Enable Http Traffic on port 80
	3) Enable Https Traffic on port 443
Steps to deploy app on EC2 instance (Make sure required protocols and ports are added to security group):

1) Place dataset folder and application code in same folder.

2) Install nginx

sudo apt-get install nginx

sudo systemctl start nginx
sudo systemctl enable nginx

sudo vi /etc/nginx/sites-available/default

Add the following code at the top of the file (below the default comments)

upstream movierecommender {
    server 127.0.0.1:8000;
}

Add a proxy_pass to movierecommender at location /

location / {
    proxy_pass http://movierecommender;
}

Restart Nginx

sudo systemctl restart nginx

3) Run the application using Python.

4) Ping http://<SERVER_IP>:<PORT_NUMBER>/get_recommendations/<user_id> to get movie recommendations

Steps to configure Load Balancer.

1) Create 2 instances as mentioned above and check if individual servers are returning correct responses.

2) Create Target group and add both instances to the target group

3) Attach a load balancer to the target group.

4) Ping the load balancer IP to get the movie recommendation