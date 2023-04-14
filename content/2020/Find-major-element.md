---
layout: blog
title: Algorithm Find major element
date: 2020-10-19 23:11:56
tags:
---

# 0. abstract

Let A[1...n] be an integer sequence, then a is called `major element` if integer a in A appears more than n/2 rounded down.

# 1. analysis

> notice that after removing two different elements in the original sequence, a major element in the old sequence is still a major element in the new sequence.

Set the counter to 1, and let `c` = A[1]. Scan the elements starting from A[2] element-wise, if the scanned element is equal to `c`, the counter adds 1, otherwise the counter minus 1. If all the elements are scanned and the counter greater than 0, then return `c` as a candidate of a major element. If the counter is 0 when `c` is compared with a[j] (1 < j < n), then call the candidate process recursively in A[j+1...n].

If the candidate `c` is greater than n/2 rounded down, it becomes major element.

# 2. code implementation

Store the sequence to file.txt

$ cat file.txt
\> 1 3 2 3 3 4 3

```c
#include <stdio.h>
#define column 7 // length of the sequence file
```

```c
int candidate(int m, int A[]){
    int j = m, c = A[m], count = 1;
    while (j<column && count>0) {
        j = j+1;
        if(A[j]==c)
            count++;
        else
            count--;
    }
    if(j==column)
        return c;
    else
        return candidate(j+1, A);
}
```

```c
int main(int argc, char *argv[])
{
    int i, j, c=0, count = 0, data[column];

    /*
     * read the data stored in file.txt
     * store the data in array data[column]
     */
    FILE *fp = fopen("./file.txt","r");
    for (j = 0; j < column; j++) {
        fscanf(fp, "%d ", &data[j]);
    }

    c = candidate(0, data);
    for (i = 0; i < column; i++) {
        if(data[i] == c) count ++;
    }

    if(count > (int)(column/2))
        printf("%d\n", c);
    else
        printf("%s\n", "no major element");

    return 0;
}

```

# Algorithm result

$ ./major\_element
\# 1 3 2 3 3 4 3
\> 3
