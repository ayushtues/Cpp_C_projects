#include <unistd.h>
#include <sys/types.h>
#include <sys/ptrace.h>
#include "linenoise.h"
#include <sys/wait.h>
#include <vector>
#include <sstream>
#include <iostream>
#include "debugger.hpp"
#include <sys/personality.h>
#include "breakpoint.hpp"

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

    if(is_prefix(command, "cont")) // if input has a "cont" prefix, contine execution
    {
        continue_execution();
    }
    else if (is_prefix(command, "break")){
        std::string addr {args[1], 2}; // naively assume that the user has written 0xADDRESS, so skip first two chars
        set_breakpoint_at_address(std::stol(addr, 0, 16));
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

void debugger::set_breakpoint_at_address(std::intptr_t addr){
    std::cout << "Set breakpoint at address 0x" << std::hex << addr << std::endl;
    breakpoint bp (m_pid, addr); // create a new breakpoint object
    bp.enable(); // enable it
    m_breakpoints[addr] = bp; // store the breakpoint-address pair in hashtable for lookup
}

void execute_debugge( const std::string &prog_name){
    if (ptrace(PTRACE_TRACEME, 0, 0, 0) < 0){
        std::cerr << "Error in ptrace\n";
        return;
    }

    execl(prog_name.c_str(), prog_name.c_str(), nullptr);

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
        personality(ADDR_NO_RANDOMIZE);
        execute_debugge(prog);

    }

    else if (pid>=1) // pid = id of child
    {
        // We are in the parent process, execute debugger
        std::cout << "Started Debugging process" << pid << '\n';
        debugger dbg(prog, pid);
        dbg.run();
    }

}