#include <stdio.h>
#include <sys/socket.h>
#include <unistd.h>
#include <stdlib.h>
#include <netinet/in.h>
#include <string.h>

#define PORT 8080

int main(int argc, char const* argv[])
{   
    int server_fd, new_socket;
    long valread;
    struct sockaddr_in address;
    int addrlen = sizeof(address);

    // construct the response, with appropriate headers and the actual message, oh and don't forget the extra blank line seperating the two
    char *hello = "HTTP/1.1 200 OK\nContent-Type: text/plain\nContent-Length: 17\n\nHello from server";

    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) // create a socket, with domain-> IP, virtual circuit service
    {
        perror("In socket");
        exit(EXIT_FAILURE);
    }

    address.sin_family = AF_INET; // the adress family, here IP
    address.sin_port = htons(PORT); // the port number 
    address.sin_addr.s_addr = INADDR_ANY; // the adress of the socket, here, our IP address

    memset(address.sin_zero, '\0', sizeof(address.sin_zero));

    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0 ) // bind socket to address
    {
        perror("In bind");
        exit(EXIT_FAILURE);   
    }

    if (listen(server_fd, 10) < 0) // set the socket to listen, have atmax 10 pending requests
    {
        perror("In listen");
        exit(EXIT_FAILURE);
    }

    while(1)
    {
        printf("\n+++++++ Waiting for new connection ++++++++\n\n");

        if((new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t* )&addrlen))< 0) // accept incoming requests and make that a new socket
        {
            perror("In accept");
            exit(EXIT_FAILURE);
        }

        char buffer[30000] = {0};
        valread = read(new_socket, buffer, 30000); // read the message sent in a buffer
        printf("%s\n", buffer);
        write(new_socket, hello, strlen(hello)); // write a reply
        printf("------------------Hello message sent-------------------\n");
        close(new_socket); // close the socket

    }

    return 0;

}