#ifndef MINIDBG_DEBUGGER_HPP
#define MINIDBG_DEBUGGER_HPP

#include <utility>
#include <string>
#include <linux/types.h>
#include <unordered_map>
#include "breakpoint.hpp"
#include "dwarf/dwarf++.hh"
#include "elf/elf++.hh"
#include <fcntl.h>

namespace minidbg {
    class debugger {
    public:
        debugger (std::string prog_name, pid_t pid)
            : m_prog_name{std::move(prog_name)}, m_pid{pid} {
        
        auto fd = open(m_prog_name.c_str(), O_RDONLY); // open the ELF executable containing the DWARF

        m_elf = elf::elf{elf::create_mmap_loader(fd)}; // map the file into memory directly, instead of reading it one bit at a time
        m_dwarf = dwarf::dwarf{dwarf::elf::create_loader(m_elf)};
        }
        void run();
        void set_breakpoint_at_address(std::intptr_t addr);

    private:
        void handle_command(const std::string& line);
        void continue_execution();        
        void dump_registers();
        uint64_t read_memory(uint64_t address);
        void write_memory(uint64_t address, uint64_t value);
        uint64_t get_pc();
        void set_pc(uint64_t pc);
        void step_over_breakpoint();
        void wait_for_signal();
        dwarf::die get_function_from_pc(uint64_t pc);
        dwarf::line_table::iterator get_line_entry_from_pc(uint64_t pc);
        void initialise_load_address();
        void print_source(const std::string& file_name, unsigned line, unsigned n_lines_context=2);
        uint64_t offset_load_address(uint64_t addr);
        siginfo_t get_signal_info();
        void handle_sigtrap(siginfo_t info);
        
        std::string m_prog_name;
        pid_t m_pid;
        uint64_t m_load_address;
        std::unordered_map<std::intptr_t, breakpoint> m_breakpoints; // to store addresses and corresponding breakpoints
        dwarf::dwarf m_dwarf;
        elf::elf m_elf;
    };
}

#endif