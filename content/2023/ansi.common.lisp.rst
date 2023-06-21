ANSI.Common.Lisp
################

:date: 2023-06-14 20:35

.. contents::

1. Introduction
===============

John McCarthy, 1958

designed to evolve

This introduction may seem a collection of grand and possibily meaningless
claims.

1.1. New Tools
--------------

Reward: you will feel as suffocated programming in C++ as an experienced C++
programmer would feel programming in Basic.

1.2. New Techniques
-------------------

extensible, reusable, rapid prototyping

1.3. A New Approach
-------------------

- interactive environments
- garbage collection
- run-time typing
- ...

2. Welcome to Lisp
==================

2.1. form
---------

*toplevel*
  interactive front-end

(+ 2 3) prefix notation

(operator argument1, argument2, ...)

2.2. evaluation
---------------

when lisp evaluates a function call like (+ 2 3)

- arguments are evaluated, from left to right
- arguments are passed to the function named by the operator

lisp provides the quote as a way of *protecting* expressions from evaluation

.. code-block:: lisp

   > (quote (+ 3 5))
   > '(+ 3 5)

return the singly taken argument verbatim

2.3. data
---------

- integer - a series of digits
- string - characters surrounded by double-quotes
- **symbols** - words, ordinarily converted to uppercase. To refer it, quote it.
- **lists** - (*list* arg1 arg2 ...) build lists
- empty list - () or NIL

2.4. list operation
-------------------

*cons* builds lists. If its 2nd argument is a list, a new list with the 1st
argument prepended to the front.

.. code-block:: lisp

   > (cons 'a '(b c d))
   (A B C D)

*car* gets the first element, *cdr* for the rest

.. code-block:: lisp

   > (car '(a b c))
   A
   > (cdr '(a b c))
   (B C)

For instance, to get 3rd element

.. code-block:: lisp

   > (car (cdr (cdr '(a b c d))))
   C
   > (third '(a b c d))
   C

2.5. truth
----------

*listp* returns true, symbol *t*, if its argument is a list

Common Lisp *predicates* often have names end with p.

.. code-block:: lisp

   > (null nil) ; *null* returns true of the empty list
   T
   > (not nil) ; *not* returns true if its argument is false
   T

(*if* *test* *then* *[else]*)

.. code-block:: lisp

   > (if (listp '(a b c))
	(+ 1 2)
	(+ 5 6))
   3

if *else* is omitted, it defaults to *nil*

.. code-block:: lisp

   > (if (listp 24)
	(+ 3 4))
   NIL

Everything except for *nil* represents *true*.

.. code-block:: lisp

   > (if 24 4 5)
   4

*and* and *or* resemble conditionals.

.. code-block:: lisp

   > (and 1 2 3) ; and
   3

   > (or 4 5) ; and or are macros
   4

2.6. functions
--------------

.. code-block:: lisp

   > (defun sum-greater (x y z)
	(> (+ x y) z))
   SUM-GREATER
   > (sum-greater 1 4 3)
   T

2.7. recursion
--------------

.. code-block:: lisp

   > (defun my-member (obj lst)
	(if (null lst)
	NIL
	(if (eql obj (car lst))
	lst
	(my-member obj (cdr lst)))))

2.8. reading lisp
-----------------

read code by indentation, with an editor supporting matching parentheses.

2.9. input and output
---------------------

  (*format* arg1 arg2 args)

  - arg1 - where the output is to be printed
    - t - output is sent to default place, toplevel
  - arg2 - string template
    - ~A indicates a position to be filled, ~% is a newline
  - args - to be inserted into template

.. code-block:: lisp

   > (format t "~A plus ~A equals ~A.~%" 2 3 (+ 2 3))
   2 plus 3 equals 5
   NIL  ; returned by the call to format

2.10. variables
---------------

*set* allows to introduce new *local* variables

.. code-block:: lisp

   > (let ((*variable* *expression*) ... )
	...)

   > (defun ask-number()
	(format t "please enter a number")
	(let ((var (read)))
	   (if (numberp var)
	     var
	     (ask-number)))

give a symbol and a value to *defparameter*

or define global constants by *defconstant*

.. code-block:: lisp

   > (defparameter *glob* 99)
   *GLOB*
   > (defconstant limit (+ *glob* 1))
   LIMIT

   > (boundp '*glob*)
   T

2.11. assignment
----------------

.. code-block:: lisp

   > (setf *glob* 37)
   *GLOB*
   > (let ((n 10))
	(setf n 2)
	n)
   > (setf a b
	   c d)  ; equals two assignments respectively

2.12. funcional programming
---------------------------

It means writing programs that work by returning values, instead of by modifying
things, which is the dominant paradigm in Lisp.

.. code-block:: lisp

   > (setf lst '(a b c a d))
   (A B C A D)
   > (remove 'a lst)
   (B C D)
   > (setf lst (remove 'a lst))  ; the original lst remains untouched
   (B C D)

2.13. iteration
---------------

.. code-block:: lisp

   > (defun show-squares (start end)
	(do ((i start (+ i 1)))  ; (variable initial update)
	 ((> i 10) 'done)  ; (stop criteria, expression evaluated when stop)
	 (format t "~A ~A~%" i (* i i))))  ; body of the loop

   > (show-squares 2 4)
   2 4
   3 9
   4 16
   DONE

.. code-block:: lisp

   > (defun show-squares (i end)
	(if (> i end)
	    'done
	    (progn  ; evaluates expressions in order, return the value of last
		(format t "~A ~A~%" i (* i i))
		(show-squares (+ i 1) end))))

*dolist* takes an argument of the form (*variable* *expression*), followed by a
body of expressions. The body will be evaluated with *variable* bound to
successive elements of the list returned by *expression*

.. code-block:: lisp

   > (defun our-length (lst)
	(let ((len 0))
	  (dolist (obj lst)  ; (variable expression
	    (setf len (+ len 1)))  ; body of expressions
	len))

.. code-block:: lisp

   > (defun our-length (lst)
	(if (null lst)
	    0
	    (+ (our-length (cdr lst)) 1)))

2.14. functions as objects
--------------------------

function that takes a function as an argument is *apply*

.. code-block:: lisp

   > (apply #'+ '(1 2 3))
   6
   > (apply #'+ 1 2 '(3 4 5))  ; any number of arguments, so long as the last is a list
   15
   ; funcall does the same thing but does not need the arguments to be
   packaged in a list
   > (funcall #'+ 1 2 3)
   6

   > (funcall #'(lambda (x) (+ x 1))
		1)
   2

2.15. types
-----------

cl types from a hierachy of subtypes and supertypes.
27 is of type *fixnum, integer, rational, real, number, atom, and t*, in order of increasing generality.

.. code-block:: lisp

   > (typep 27 'integer)
   T

2.16. looking forward
---------------------

So far we have barely scratched the surface of Lisp.

- interactive on toplevel
- prefix syntax means any number of arguments
- parentheses are not an issue, we use indentation
- funcional programming, which avoid side-effects, is the dominant paradigm

2.17. exercise
--------------

#. describe what happens when the following expressions are evaluated

   - (+ (- 5 1) (+ 3 7))  ; 14
   - (list 1 (+ 2 3))  ; (1 5)
   - (if (listp 1) (+ 1 2) (+ 3 4))  ; 7
   - (list (and (listp 3) t) (+ 1 2))  ; (nil 3)

#. give three distinct *cons* expressions that return (a b c)

   - (cons 'a '(b c))
   - (cons 'a (cons 'b '(c)))
   - I can't think of another solution...

#. using *car* and *cdr*, define a function to return the fourth element of a list

   .. code-block:: lisp

      (defun my-fourth (lst)
         (car (cdr (cdr (cdr lst)))))

#. define a function that takes two arguments and returns the greater of the two

   .. code-block:: lisp

      (defun my-greater (x y)
         (if (> x y)
	     x
	     y))

#. what do these functions do

   .. code-block:: lisp

      (defun enigma (x)
         (and (not (null x))
	      (or (null (car x))
	          (enigma (cdr x)))))

      ; if nil is in x

   .. code-block:: lisp

      (defun mystery (x y)
         (if (null y)
	     nil
	     (if (eql (car y) x)
	     0
	     (let ((z (mystery x (cdr y))))
	          (and z (+ z 1))))))

      ; count number of elements in y which differ x

#. what could occur in place of the *x* in each of the following exchanges

   - > (car (**car** (cdr '(a (b c) d))))
     B
   - > (**or** 13 (/ 1 0))
     13
   - > (**apply** #'list 1 nil)
     (1)

#. using only operators introduced in this chapter, define a function that takes a list as an argument and returns true if one of its elements is a list

   .. code-block:: lisp

      (defun if-have-list (lst)
         (if (null lst)
           nil
           (if (listp (car lst))
             t
             (if-have-list (cdr lst)))))

#. give iterative and recursive definitions of a function that

   (1) takes a positive integer and prints that many dots

   .. code-block:: lisp

      (defun print-dots-iteratively (x)
         (do ((i 0 (+ i 1)))
	     ((eql i x) 'done)
	     (format t ".")
	 )
      )

   .. code-block:: lisp

      (defun print-dots-recursively (x)
         (if (eql x 0)
	     'done
	     (progn
	        (format t ".")
		(print-dots-recursively (- x 1)))))

   (2) takes a list and returns the number of times the symbol *a* occurs in it

   .. code-block:: lisp

      (defun a-interative-counter (lst)
          (let ((count 0))
	     (dolist (obj lst)
	        (if (eql obj 'a)
	           (setf count (+ count 1))))
	      count))

   .. code-block:: lisp

      (defun a-recursive-counter (lst)
         ()
      )

#. a friend is trying to write a function that returns the sum of all the non-nil elements in a list. he has written two versions of this function, and neither of them work. explain what's wrong with each, and give a correct version

   (a)
       .. code-block:: lisp

	  (defun summit (lst)
	     (remove nil lst)
	     ; (setf lst (remove nil lst)) to update lst
	     (apply #'+ lst))

   (b)
       .. code-block:: lisp

	  (defun summit (lst)
	     (let ((x (car lst)))
	        (if (null x)
		    ; I strongly doubt here goes wrong
		    ; (cdr last-element) is still nil
		    ; so this is an infinite loop
		    (summit (cdr lst))
		    (+ x (summit (cdr lst))))))


3. Lists
========

3.1. conses
-----------

.. code-block:: lisp

   > (setf x (cons 'a nil))
   > (setf y (list 'a 'b 'c))  ; flat list
   > (setf z (list 'a (list 'b 'c) 'd))  ; nested list

   (defun my-listp (x)
       (or (null x) (consp x)))

   (defun my-atom (x)
       (not (consp x)))

*nil* is both an *atom* and *list*

3.2. equality
-------------

calling *cons* makes lisp allocate a piece of memory for two pointers, so calling *cons* twice generates two distinctly different objects.

.. code-block:: lisp

   > (eql (cons 'a nil) (cons 'a nil))
   NIL

*eql* returns true only if **the same object**, and *equal* only needs printed result being same.
