---
title: linux system call
date: 2026-03-07 05:59
tags:
---

[TOC]

test

road map:
write -> glibc -> syscall -> syscall asm
-> TRAP to [MSR]Model Specific Register
-> entry_SYSCALL_64
-> do_syscall_64
-> syscall table
-> sys_write ( write in kernel)
-> diy

# 0. intro

# 1. cpu priviledge level

- ring 0   <--->   kernel
- ring 1
- ring 2
- ring 3   <--->   user app

# 2. user/kernel mode

## 2.1 user mode

limited access to hardware and system resource

- isolated process memory space

- access hardware not directly but only through **syscall**

## 2.2 kernel mode

full access to hardware and system resource

- no memory space isolation

1. save user mode context (PC, regs)
2. switch to kernel mode
3. jump to pre-registered handler

# 3. CPU events

## 3.1 interrupt

external hardward, asynchronous

- timer
- IO (keyboard, network, storage)

## 3.2 exception

synchronous

### 3.2.1 fault

execute the instruction **again**

- page fault

- protection fault

- x / 0

### 3.2.1 trap

usually deliberately, execute **next** instruction

- system call

user mode is transitioned into kernel mode through **TRAP** if OS services,
such as file io, process creation, network communication and memory allocation
are needed.

# 4. system call

## 4.1 ancient system calls

Linux kernel preserves a special code `0x80` which corresponds `ia32_syscall`.

[trap_init](https://github.com/torvalds/linux/blob/v3.13/arch/x86/kernel/traps.c#L770)
```C
void __init trap_init(void)
{
        /* ..... other code ... */

        set_system_intr_gate(IA32_SYSCALL_VECTOR, ia32_syscall);
```
where `IA32_SYSCALL_VECTOR` is 0x80, defined in
[irq_vectors.h](https://github.com/torvalds/linux/blob/v3.13/arch/x86/kernel/traps.c#L770)

Since **only one** exception code is preserved, to let kernel know which system
call to execute.

Userspace app place syscall code into `eax`, and args into other regs.
[ia32entry.S](https://github.com/torvalds/linux/blob/v3.13/arch/x86/ia32/ia32entry.S#L378-L397)

```
 * Emulated IA32 system calls via int 0x80.
 *
 * Arguments:
 * %eax System call number.
 * %ebx Arg1
 * %ecx Arg2
 * %edx Arg3
 * %esi Arg4
 * %edi Arg5
 * %ebp Arg6    [note: not saved in the stack frame, should not be touched]
 *
```

### 4.1.1. issue a legacy system call in asm

`exit` system call has one arg: return value.

[syscall_32.tbl](https://github.com/torvalds/linux/blob/v3.13/arch/x86/syscalls/syscall_32.tbl)

```
1 i386  exit      sys_exit
```

Place the syscall number, `1` here, into `eax`, and the first arg (return value)
to `ebx`.

```C
int
main(int argc, char *argv[])
{
  unsigned int syscall_nr = 1; // exit syscall number
  int exit_status = 42; // exit(int status), 42 here.

  asm ("movl %0, %%eax\n"
       "movl %1, %%ebx\n"
       "int $0x80" // issue exception here.
    : /* output parameters */
      /* (none) */
    : /* input parameters mapped to %0 and %1, repsectively */
      // "m" memory operation number, gcc place them into reg/mem
      "m" (syscall_nr), "m" (exit_status)
    : /* registers that we are changing, unneeded since we are calling exit */
      "eax", "ebx");
}
```

```
$ gcc -o test test.c
$ ./test
$ echo $?
42
```

## 4.2. fast system call

32bit: sysenter and sysexit

64bit: `syscall` and `sysret`

> [man 2 syscall](https://man.archlinux.org/man/syscall.2)
> 
> syscall() is a small library function that invokes the system call
> whose assembly language interface has the specified number
> with the specified arguments.
> Employing syscall() is useful, for example,
> when invoking a system call that has no **wrapper function** in the C library.
> 
> syscall() saves CPU registers before making the system call,
> restores the registers upon return from the system call,
> and stores any error returned by the system call in errno(3).

>[man 2 syscalls](https://man.archlinux.org/man/syscalls.2)
>
> System calls are generally not invoked directly, but rather via **wrapper**
> **functions in glibc** (or perhaps some other library).
>
> Often, but not always, the name of the wrapper function is the same
> as the name of the system call that it invokes.
> For example, glibc contains a function chdir() which invokes the 
> underlying "chdir" system call.
> 
> Often the glibc wrapper function is quite thin, doing little work other than
> copying arguments to the right registers before invoking the system call,
> and then setting errno appropriately after the system call has returned.

still syscall `exit`, [syscall_64.tbl](https://github.com/torvalds/linux/blob/v3.13/arch/x86/syscalls/syscall_64.tbl#L69)

```
60      common  exit                    sys_exit
```

the syscall number `60` goes to `rax`, and first arg to `rdi`

### 4.2.1 assembly way

```C
int main(int argc, char *argv[])
{
  unsigned long syscall_nr = 60;
  long exit_status = 42;

  asm ("movq %0, %%rax\n"
       "movq %1, %%rdi\n"
       "syscall"
    : /* output parameters, we aren't outputting anything, no none */
      /* (none) */
    : /* input parameters mapped to %0 and %1, repsectively */
      "m" (syscall_nr), "m" (exit_status)
    : /* registers that we are "clobbering", unneeded since we are calling exit */
      "rax", "rdi");
}
```

```
$ gcc -o test test.c
$ ./test
$ echo $?
42
```

### 4.2.2 easier, glibc wrapper way

```C
#include <unistd.h>

int
main(int argc, char *argv[])
{
  unsigned long syscall_nr = 60;
  long exit_status = 42;

  syscall(syscall_nr, exit_status);
}
```

```
$ gcc -o test test.c
$ ./test
$ echo $?
42
```

## 4.3 syscall chain

[musl](https://www.musl-libc.org/)

### 4.3.1 musl libc write

```C
#include <unistd.h>

int main() {
    write(1, "hello\n", 6);
    return 0;
}
```

[the musl write implementation](https://git.musl-libc.org/cgit/musl/tree/src/unistd/write.c) is easier

```C
ssize_t write(int fd, const void *buf, size_t count)
{
        return syscall_cp(SYS_write, fd, buf, count);
}
```

### 4.3.2 syscall_cp

[syscall_cp](https://git.musl-libc.org/cgit/musl/tree/src/internal/syscall.h) is a macro

```C
#define syscall_cp(...) __syscall_ret(__syscall_cp(__VA_ARGS__))
```

### 4.3.3 __syscall

[__syscall](https://git.musl-libc.org/cgit/musl/tree/arch/x86_64/syscall_arch.h)

```C
static __inline long __syscall6(long n, long a1, long a2, long a3, long a4, long a5, long a6)
{
	unsigned long ret;
	register long r10 __asm__("r10") = a4;
	register long r8 __asm__("r8") = a5;
	register long r9 __asm__("r9") = a6;
	__asm__ __volatile__ ("syscall" : "=a"(ret) : "a"(n), "D"(a1), "S"(a2),
						  "d"(a3), "r"(r10), "r"(r8), "r"(r9) : "rcx", "r11", "memory");
	return ret;
}
```

### 4.3.4 to be continued

Should I go for a PhD degree ?