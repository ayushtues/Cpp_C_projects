#ifndef MINIDBG_BREAKPOINT_HPP
#define MINIDBG_BREAKPOINT_HPP

#include <cstdint>
#include <sys/ptrace.h>
#include <unistd.h>

namespace minidbg{
    class breakpoint{
    public:
        breakpoint() = default;
        breakpoint(pid_t pid, std::intptr_t addr)
            : m_pid{pid}, m_addr{addr}, m_enabled{false}, m_saved_data{}
        {}

        void enable(){
            auto data = ptrace(PTRACE_PEEKDATA, m_pid, m_addr, nullptr); // get data at current instruction address
            m_saved_data = static_cast<uint8_t>(data & 0xff);  // get the bottom bytes of the data
            uint64_t int3 = 0xcc; // the interrupt instruction in x86
            uint64_t data_with_int3 = ((data & ~0xff) | int3); // data & ~0xff clears the bottom bytes and then OR with int3 sets the interrupt
            
            ptrace(PTRACE_POKEDATA, m_pid, m_addr, data_with_int3); // overwrite the memory with the new data

            m_enabled = true; // set breakpoint as enabled
        }

        void disable(){
            auto data = ptrace(PTRACE_PEEKDATA, m_pid, m_addr, nullptr); // read data at current execution point
            
            auto restored_data = ((data & ~0xff) | m_saved_data); // since ptrace reads entire word, we need to get 
                                                                //the lowermost bytes specifically and set them with the saved value
            
            ptrace(PTRACE_POKEDATA, m_pid, m_addr, restored_data); // overwrite memory 
            
            m_enabled = false;
        }

        auto is_enabled() const -> bool {return m_enabled; }
        auto get_address() const -> std::intptr_t {return m_addr;}

    private:
        pid_t m_pid;
        std::intptr_t m_addr;
        bool m_enabled;
        uint8_t m_saved_data; // data which used to be at breakpoint address before we interrupted it, for later execution    

    };
}

#endif
