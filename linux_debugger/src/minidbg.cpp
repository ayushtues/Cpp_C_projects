#include <unistd.h>
#include <sys/types.h>
#include <sys/ptrace.h>
#include "linenoise.h"
#include <sys/wait.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "debugger.hpp"
#include <sys/personality.h>
#include "breakpoint.hpp"
#include "registers.hpp"
#include <iomanip>
#include <fcntl.h>
#include <map>

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
    else if (is_prefix(command, "register")){ // read/write to/from registersm or dump them
        if(is_prefix(args[1], "dump")){
            dump_registers();
        }
        else if (is_prefix(args[1], "read")){
            std::cout << get_register_value(m_pid, get_register_from_name(args[2])) << std::endl;
        }
        else if (is_prefix(args[1], "write")){
            std::string val {args[3], 2}; // assume 0xVAL
            set_register_value(m_pid, get_register_from_name(args[2]), std::stol(val, 0, 16));
        }
    }

    else if (is_prefix(command, "memory")){ // read/write to/from memory

        std::string addr {args[2], 2}; // assume 0xADDRESS

        if(is_prefix(args[1], "read")){
            std::cout<< std::hex << read_memory(std::stol(addr, 0, 16))<<std::endl;
        }
        if(is_prefix(args[1], "write")){
            std::string val {args[3], 2}; // assume 0xVAL
            write_memory(std::stol(addr, 0, 16), std::stol(val, 0, 16));
        }
    }

    else if (is_prefix(command, "stepi"))
    {
        single_step_instruction_with_breakpoint_check();
        auto line_entry = get_line_entry_from_pc(get_pc());
        print_source(line_entry->file->path, line_entry->line);
    }

    else if(is_prefix(command, "step"))
    {
        step_in();
    }

    else if(is_prefix(command, "next"))
    {
        step_over();
    }

    else if(is_prefix(command, "finish"))
    {
        step_out();
    }

    else{
        std::cerr << "Unknown command\n";
    }
}

void debugger::continue_execution(){
    step_over_breakpoint();
    ptrace(PTRACE_CONT, m_pid, nullptr, nullptr); // continue the process using ptrace
    wait_for_signal();

}

void debugger::run(){

    int wait_status;
    initialise_load_address();
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

void debugger::dump_registers(){
    for ( const auto& rd : g_register_descriptors){
        std::cout << rd.name <<"0x"<<std::setfill('0')<<std::setw(16)<<std::hex<<get_register_value(m_pid, rd.r)<<std::endl;
    }
}

// functions to read/write to/from memory
uint64_t debugger::read_memory(uint64_t address){
    return ptrace(PTRACE_PEEKDATA, m_pid, address, nullptr);
}

void debugger::write_memory(uint64_t address, uint64_t value){
    ptrace(PTRACE_POKEDATA, m_pid, address, value);
}

// get/set value of Program Counter
uint64_t debugger::get_pc(){
    return get_register_value(m_pid, reg::rip);
}

void debugger::set_pc(uint64_t pc){
    set_register_value(m_pid, reg::rip, pc);
}

void debugger::step_over_breakpoint(){

    if (m_breakpoints.count(get_pc())){
        auto &bp = m_breakpoints[get_pc()]; // check if there is a breakpoint at the value of the current PC

        if(bp.is_enabled()){
            bp.disable(); // disable the breakpoint
            ptrace(PTRACE_SINGLESTEP, m_pid, nullptr, nullptr); // execute the original instruction
            wait_for_signal();
            bp.enable(); // reenable breakpoint
        }
    }

}

void debugger::wait_for_signal(){
    int wait_status;
    auto options = 0;
    waitpid(m_pid, &wait_status, options);

    auto siginfo = get_signal_info();

    switch (siginfo.si_signo){
    case SIGTRAP:
        handle_sigtrap(siginfo);
        break;
    case SIGSEGV:
        std::cout << "Yay, segfault. Reason: " << siginfo.si_code << std::endl;
        break;    
    default:
        std::cout<<"Got signal " << strsignal(siginfo.si_signo) << std::endl;
    }

}

dwarf::die debugger::get_function_from_pc(uint64_t pc){
    for (auto &cu : m_dwarf.compilation_units()){ // loop through compilation units
        if (die_pc_range(cu.root()).contains(pc)){ // if pc is found in the range
            for (const auto& die : cu.root()){
                if ( die.tag == dwarf :: DW_TAG::subprogram){ //loop across subprograms/functions
                    if(die_pc_range(die).contains(pc)) // if pc is found
                    {
                        return die; 
                    }
                }
            }
        }
    }

    throw std::out_of_range{"Cannot find function"};
}

dwarf::line_table::iterator debugger::get_line_entry_from_pc(uint64_t pc){

    for (auto &cu : m_dwarf.compilation_units()){ // loop over compilation units
        if( die_pc_range(cu.root()).contains(pc)){ // if it contains PC
            
            auto &lt = cu.get_line_table(); // get line table
            auto it = lt.find_address(pc); // find the pc in the line table
            
            if( it == lt.end()){
                throw std::out_of_range{"Cannot find line entry"};
            }
            
            else{
                return it;
            }
        }
    }
}

// get the initial load address and store it in a variable
void debugger::initialise_load_address(){

    // if this is a dynamic library ( e.g. PIE)
    if (m_elf.get_hdr().type == elf::et::dyn){

        std::ifstream map("/proc/" + std::to_string(m_pid) + "/maps"); 

        // Read the first address from the file
        std::string addr;
        std::getline(map, addr, '-');

        m_load_address = std::stoi(addr, 0, 16); 
    }
}

// caculate the offset of the current address
uint64_t debugger::offset_load_address(uint64_t addr){
    return addr - m_load_address;
}

void execute_debugge( const std::string &prog_name){
    if (ptrace(PTRACE_TRACEME, 0, 0, 0) < 0){
        std::cerr << "Error in ptrace\n";
        return;
    }

    execl(prog_name.c_str(), prog_name.c_str(), nullptr);

}

// print the source code around the requested line ( the one we are inspecting via a breakpoint )
void debugger::print_source( const std::string& file_name, unsigned line, unsigned n_lines_context){

    std::ifstream file{file_name};

    // get a context window of neigbouring lines around the desired line
    auto start_line = line <= n_lines_context ? 1 : line - n_lines_context;
    auto end_line = line + n_lines_context + ( line < n_lines_context ? n_lines_context - line : 0 ) + 1;

    char c{};
    auto current_line = 1u;

    // Skip lines up until start_line
    while(current_line != start_line && file.get(c)){
        if ( c == '\n'){
            ++current_line;
        }
    }

    // Output cursor if we're at the current line
    std::cout << (current_line == line ? "> " : " ");

    // Write lines up until end_line
    while (current_line <= end_line && file.get(c)) {
        std::cout << c;
        if(c == '\n'){
            ++current_line;
            // Output cursor if we're at the current line
            std::cout << (current_line==line ? "> " : " ");
        }
    }

    // Write newline and make sure that the stream is flushed properly
    std::cout << std::endl;
}

siginfo_t debugger::get_signal_info(){
    siginfo_t info;
    ptrace(PTRACE_GETSIGINFO, m_pid, nullptr, &info); // gets info about the last signal
}

void debugger::handle_sigtrap(siginfo_t info){
    switch (info.si_code){

        // one of these will be set if a breakpoint was hit
        case SI_KERNEL:
        case TRAP_BRKPT:
        {
            set_pc(get_pc() -1); // put the pc back to where it should be   
            std::cout << "Hit breakpoint at address 0x"<< std::hex << get_pc() << std::endl;

            auto offset_pc = offset_load_address(get_pc()); // remember to offset the pc for querying DWARF
            auto line_entry = get_line_entry_from_pc(offset_pc);
            
            print_source(line_entry->file->path, line_entry->line); // print the source code around the breakpoint
            return;
        }   
        
        // this will be set if the signal was sent by single stepping
        case TRAP_TRACE:
            return;
        
        default:
            std::cout<< "Unkown SIGTRAP code " << info.si_code << std::endl;
            return;

    }
}

void debugger::single_step_instruction(){
    ptrace(PTRACE_SINGLESTEP, m_pid, nullptr, nullptr);
    wait_for_signal();
}

void debugger::single_step_instruction_with_breakpoint_check(){
    // first, check to see if we need to disable and enable a breakpoint
    if(m_breakpoints.count(get_pc())){
        step_over_breakpoint();
    }
    else{
        single_step_instruction();
    }
}

void debugger::step_out(){
    
    // read the frame pointer and read a word of memory at the retyrb address which is + 8 of the start of the stack frame
    auto frame_pointer = get_register_value(m_pid, reg::rbp);
    auto return_address = read_memory(frame_pointer+8);

    bool should_remove_breakpoint = false;

    if(!m_breakpoints.count(return_address))
    {
        set_breakpoint_at_address(return_address); // set a breakpoint at the address if not already
        should_remove_breakpoint = true;
    }

    continue_execution(); // continue execution

    if(should_remove_breakpoint)
    {
        remove_breakpoint(return_address); // remove the previously set breakpoint
    }

}

void debugger::remove_breakpoint(std::intptr_t addr)
{
    if(m_breakpoints.at(addr).is_enabled())
    {
        m_breakpoints.at(addr).disable();
    }

    m_breakpoints.erase(addr);
}

void debugger::step_in(){

    auto line = get_line_entry_from_pc(get_offset_pc())->line;

    // keep stepping over instructions until you get a new line
    while (get_line_entry_from_pc(get_offset_pc())->line == line)
    {
        single_step_instruction_with_breakpoint_check();
    }

    auto line_entry = get_line_entry_from_pc(get_offset_pc());
    print_source(line_entry->file->path, line_entry->line);
}

uint64_t debugger::get_offset_pc(){
    return offset_load_address(get_pc());
}


// offset addresses from DWARF info by the load address
uint64_t debugger::offset_dwarf_address(uint64_t addr)
{
    return addr + m_load_address;
}

void debugger::step_over(){

    auto func = get_function_from_pc(get_offset_pc());
    auto func_entry = at_low_pc(func); // libelfin functions to get high and low PC values, given function IDE
    auto func_end = at_high_pc(func);

    auto line = get_line_entry_from_pc(func_entry);
    auto start_line = get_line_entry_from_pc(get_offset_pc());

    std::vector<std::intptr_t> to_delete{}; // breakpoints to be deleted    

    while(line->address < func_end) // loop over line table entries
    {
        auto load_address = offset_dwarf_address(line->address);
        if(line->address != start_line->address && !m_breakpoints.count(load_address)) // make sure its not the current line and also is not already a breakpoint 
        {
            set_breakpoint_at_address(load_address);
            to_delete.push_back(load_address); // store for later deletion
        }
        ++line;
    }

    auto frame_pointer = get_register_value(m_pid, reg::rbp);
    auto return_address = read_memory(frame_pointer + 8);

    // set a breakpoint at the return address of the function
    if(!m_breakpoints.count(return_address)){
        set_breakpoint_at_address(return_address);
        to_delete.push_back(return_address);
    }

    continue_execution(); // continue till one of the breakpoints is hit

    for(auto addr : to_delete)
    {
        remove_breakpoint(addr); //remove all the temporary breakpoints we had set
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