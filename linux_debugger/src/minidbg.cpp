#include <unistd.h>
#include <sys/types.h>
#include <sys/ptrace.h>
#include "linenoise.h"
#include <sys/wait.h>
#include <vector>
#include <sstream>
#include <iostream>
#include "debugger.hpp"

using namespace minidbg;

std::vector<std::string> split(const std::string &s, char delimiter)
{
    std::vector<std::string> out{};
    std::stringstream ss {s}; // kind of creates a stream of strings like cin , add s to it
    std::string item;

    while(std::getline(ss, item, delimiter)) // get next string from the string stream
    {
        out.push_back(item);
    }

    return out;
}

bool is_prefix(const std::string &s, const std::string &of)
{
    if ( s.size() > of.size()){return false;}

    return std::equal(s.begin(), s.end(), of.begin());
}


void debugger::handle_command(const std::string &line)
{
    auto args = split(line, ' '); // split input line
    auto command = args[0]; 

    if(is_prefix(command, "continue")) // if input is "continue", contine execution
    {
        continue_execution();
    }
    else{
        std::cerr << "Unknown command\n";
    }
}

void debugger::continue_execution(){

    ptrace(PTRACE_CONT, m_pid, nullptr, nullptr); // continue the process using ptrace

    int wait_status;
    auto options = 0;
    waitpid(m_pid, &wait_status, options); // wait until signalled
}

void debugger::run(){

    int wait_status;
    auto options = 0;
    waitpid(m_pid, &wait_status, options); // wait for the child process to launch and send a SIGTRAP signal

    char *line = nullptr;
    while((line = linenoise("minidbg> ")) != nullptr) // keep on getting input until EOF
    {
        
        handle_command(line);
        linenoiseHistoryAdd(line); // Add the input line to linenoise history
        linenoiseFree(line); // Free resources 
    }

}




int main(int argc, char *argv[])
{
    if(argc < 2)
    {
        std::cerr << "Program name not specified";
        return -1;
    }

    auto prog = argv[1];

    auto pid = fork();

    if(pid==0)
    {
        // We are in the child process, execute program to be debugged

        ptrace(PTRACE_TRACEME, 0, nullptr, nullptr); // allow the parent process to trace this process
        execl(prog, prog, nullptr); // execute the program to debug
    }

    else if (pid>=1) // pid = id of child
    {
        // We are in the parent process, execute debugger
        std::cout << "Started Debugging process" << pid << '\n';
        debugger dbg(prog, pid);
        dbg.run();
    }

}