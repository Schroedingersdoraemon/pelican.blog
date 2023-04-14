---
layout: blog
title: '(updating) sed, a stream editor'
date: 2021-08-16 18:11:17
tags:
---

# 0. abstract

`sed`, a stream editor, is able to perform basic text transformation on an input text stream, which is a file or input from pipeline. The text transformation pattern performed by `sed` is called *SCRIPT* here.

# 1. running sed

## 1.1 overview

Normally `sed` is invoked like this:

```shell
sed SCRIPT INPUT_FILE...
```

For instance, to replace all "hello" with "world" in the input file:

```shell
sed 's/hello/world/' input.txt > output.txt
```

If no INPUT\_FILE is specified, or specified as `-`, then `sed` filters the contents of the standard input and stream out the transformation result as you return text to standard input.

```shell
sed 's/hello/world/' input.txt
sed 's/hello/world/' < input.txt
cat input.txt | sed 's/hello/world/'
cat input.txt | sed 's/hello/world/' - # Dylan: still not know the difference between line 3 and 4, with the same text result.
```

`sed` stream the result to standard output. With `-i` option specified, it edits the file **in-place** instead of printing to standard out. By default `sed` prints all processed input. Use `-n` to suppress output, and `p` to print specific line.

```shell
# print the 45th of the input file
sed -n '45p' input.txt
```

If `-e` or `-f` options are used to specify a *SCRIPT*, all non-option parameters are taken as input files.

```shell
sed 's/hello/world/' input.txt

sed -e 's/hello/world/' input.txt
sed --expression='s/hello/world/' input.txt

echo 's/hello/world/' > myscript.sed
sed -f myscript.sed input.txt
sed --file=myscript.sed input.txt
```

## 1.2 command-line options

The full format for invoking `sed` is:

```
sed OPTIONS... SCRIPT INPUTFILE...
```

`sed` may be invoked with the following command-like options:

|option|meaning|
|:-:|:-|
|-\-version|print out version of `sed` and copyright info |
|-\-help|briefly summerized options and bug-reporting address|
|-n<br>-\-quiet<br>-\-silent|disable automatic printign, and ony produces the output when explicitly told to via `p` command|
|-\-debug|print the input sed program in canonical form, and annotate program execution|
|-e *SCRIPT*<br>-\-expression=*SCRIPT*||
|-f *SCRIPT_FILE*<br>-\-file=*SCRIPT_FILE*||
|-i[*SUFFIX*]<br>-\-in-place[=*SUFFIX*]|This opion specifies that files are to be edited in-place. GNU `sed` implemenets this by creating a temporary file and stream output to this file. When the end of the file is reached, the temporary file is renamed to the orginal file's name.<br> *ASTERISK* is replaced with the filename.|
|-l<br>-\-line-length=*N*|Specify the default line-wrap length for `l` command. 0 means to never wrap long lines. If not specified, by default, 70 is taken.|
|-\-posix|If you want to disable extension that violates the POSIX standard, set the `POSIXLY_coRRECT` variable to a non-empty value.|
|-b<br>-\-binary|Open the input fil in binary mode and ignore distinctive platform encoding.|
|-\-follow-symlinks|`-i` specified, `sed` will follow the symlink and edit the ultimate destination of the link.|
|-E<br>-r<br>-\-regexp-extended|Use extended regular expression, use `-E` for portability.|
|-s<br>-\-separate|By default, `sed` will consider the files specified on the command line as a single continuous long stream. **TODO_MARK**|
|-\-sandbox|In sandbox mode, `e/w/r` commands are rejected|
|-u<br>-\-unbuffered|Buffer both input and output as minimally as practical. Particularly useful if the input is from the likes of `tail -f`, and you wish to see the transformed output as soon as possible.|
|-z<br>-\-null-data<br>-\-zero-terminated|Treat the input as a set of lines, each terminated by a zero byte(the ASCII 'NUL')|

## 1.3 exit status

|exit code|status|
|:-:|:-|
|0|successful completion|
|1|invalid command/syntax/regular expression, or `--posix`|
|2|One or more of the input file specified could not be opened|
|4|I/O error, or a serious processing error|




# 2. `sed` scripts

## 2.1 scripts overview

A `sed` program consists of one or more `sed` commands, passed in by one or more of the `-e`, `-f`, `--expression`, and `--file` options, or the first non-option argument if zero of there options are used. This document will refer to "the" `sed` script; this is understood to mean the in-order concatenation of all of the *SCRIPTS* and *SCRIPTS_FILE* passed in.

`sed` commands follow this syntax:

```
[addr]X[options]
```

*X* is a single-letter `sed` command. *[addr]* is an optional line address. If *[addr]* is specified, the command *X* will be executed onl on the matched lines. *[addr]* can be a single line number, a regular expression, or a range of lines. Additional *[options]* are used for some `sed` commands.

For instance, the following examples deleted lines 30 to 35 in the input. `30,35`is an address range. `d` is the delete command:

```shell
sed '30,35d' input.txt
```

For another instance, the following example prints all input until a line starting with 'foo' is found. If such line is found, `sed` will terminate with exit status 42. If such line was not found (and no other error occurred), `sed` will exit with status 0.

```shell
sed '/^foo/q42' input.txt
```

Commands with a *SCRIPT* or *SCRIPT_FILE* can be separated by semicolons or newlines(ASCII 10). Multiple scripts can be specified with `-e` or `-f`.

The following examples are all equivalent. They perform two `sed` operations: delete any lines matching the regular expression `/^foo/`, and replacing all occurrences of the string 'hello' with 'world':

```shell
sed '/^foo/d; s/hello/world/' input.txt

sed -e '/^foo/d' -e 's/hello/world/' input.txt

echo '/^foo/d' > script.sed
echo 's/hello/world/' >> script.sed
sed -f script.sed input.txt

echo 's/hello/world/' > script2.sed
sed -e '/^foo/d' -f script2.sed input.txt
```

Note that commands `a`, `c`, `i`, due to their syntax, cannot be followed by semicolons working as command separators and thus should be terminated with newlines or be placed at the end of a *SCRIPT* or *SCRIPT_FILE*. Commands can also be preceded with optional non-significant whitespace characters.

## 2.2 commands list

The following commands, some are standard POSIX commands while other are GNU extensions, are supported in GNU `sed`. Mnemonics are shown in parentheses.

|||
|:-:|:-|
|a\\<br>*TEXT*|Append *TEXT* after a line|
|a *TEXT*|Append *TEXT* after a line(alternative syntax)|
|b *LABEL*|Branch unconditionally to *LABEL*. The *LABEL* may be omitted, in which case the next cycle is started.|
|c\\<br>*TEXT*|Replace (change) lines with *TEXT*.|
|c *TEXT*|Replace (change) lines with *TEXT* (alternative syntax).|
|d|Delete the pattern space; immediately start next cycle.|
|D|If patter space contains newlines, delete text in the pattern space up to the first newline, and restart cycle with the resultant pattern space, without reading a new line of input.<br>If pattern space contains no newline, start a normal new cycle as if the `d` command was issued.|
|e|Executes the command that is found in pattern space and replaces the pattern space with output; a trailing newline is suppressed.|
|e *COMMAND*|Executes *COMMAND* and sends its output to the output stream. The command can run across multiple lines, all but the last ending with a back-slash.|
|F|(filename) Print the file name of the current input file (with a trailing newline).|
|g|Replace the contens of the pattern space with the contents of the hold space.|
|G|Append a newline to the contents of the pattern space, and then append the contents of the hold space to tthat of the pattern space.|
|h|(hold) Replace the contents of the hold space with the contents of the patter space.|
|H|Append a newline to the contents of the hold space, and then append the contents of the pattern space to that of the hold space.|
|i\\<br>*TEXT*|insert *TEXT* before a line.|
|i *TEXT*|insert *TEXT* before a line (alternative syntax).|
|l|Print the pattern space in an unambiguous form.|
|n|(next) If auto-print is not disabled, print the pattern space, then, regardless, replace the pattern space with the next line of input. If there is no more input then sed exits without processing any more commands.|
|N|Add a newline to the pattern space, then append the next line of input to the pattern space. If there is no more input then sed exits without processing any more commands.|
|p|Print the pattern space.|
|P|Print the pattern space, up to the first \<newline>.|
|q[exit-code]|(quit) Exit `sed` without processing any more commands or input|
|Q[exit-code]|(quit) Same as `q`, but will not print the contents of pattern space. Like `q`, it provides the ability to return an exit code to the caller.|
|r *FILENAME*|Reads file *FILENAME*|
|R *FILENAME*|Queue a line of *FILENAME* to be read and inserted into the output stream at the end of the current cycle, or when the next input line is read.|
|s/*REGEXP/REPLACEMENT/[flags]*|(substitute) Match the regular-expression against the content of the pattern space. If found, replace matched string with *REPLACEMENT*.|
|t *LABEL*|(test) Branch to *LABEL* only if there has ben a successful `s`ubstitution since the last input line was read or conditional branch was taken. The *LABEL* may be omitted, in which case the next cycle is started.|
|T *LABEL*|(test) Branch to *LABEL* only if there has ben no successful `s`ubstitution since the last input line was read or conditional branch was taken. The *LABEL* may be omitted, in which case the next cycle is started.|
|v *[VERSION]*|(version) This command does nothing, but makes `sed` fail if GNU `sed` extensions are not supported, or if the requested version is not available.|
|w *FILENAME*|Write the pattern space to *FILENAME*|
|W *FILENAME*|Write the pattern space to *FILENAME*|
|x|Exchange the contents of the hold and pattern spaces.|
|y/src/dst/|Transliterate any characters in the pattern space which match any of the *SOURCECHARS* with the corresponding characters in *DEST-CHARS*.|
|z|(zap) This command empties the content of patter space.|
|#|A comment, untilthe next newline.|
|{*CMD* ; *CMD* ...}|Group several commands together.|
|=|Print the current input line number (with a trailing newline)|
|: *LABEL*|Specify the location of *LABEL* for branch commands (b, t, T)|

## 2.3 the "s" command

The `s` command, as in substitute, is probably the most important in `sed` and has a lot of different options. The syntax of it is 's/*REGEXP/REPLACEMENT/FLAGS*'.

The *REPLACEMENT* can contain \n (n being a number from 1 to 0, inclusive) references, which refer to the portion of the match which is contained btween the nth \( and its matchiing \). Also, the *REPLACEMENT* can contain unescaped & which reference the **whole** matched portion of the pattern space.

As a GNU `sed` extension, you can include a special sequence made of a backslash and one of the letters L, l, U, u, or E. The meaning is as follows:

|||
|:-:|:-|
|\L|Turn the *REPLACEMENT* to lowercase until a \U or \E is found. |
|\l|Turn the next character to lowercase|
|\U|Turen the *REPLACEMENT* to uppercase until a \L or \E is found.|
|\u|Turn the next character to uppercase.|
|\E|Stop case conversion started by \L or \U|


## common commands

## other commands

## programming commands

## extended commands

## multiple commands syntax
