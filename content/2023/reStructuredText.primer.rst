reStructuredText primer
#######################
:date: 2023-04-10 17:46

0. abstract
===========

reStructuredText, or reST, is designed to be a simple yet unobtrusive markup
language.


1. basics
=========

1.1. headings
-------------

.. code-block:: rst

    Title
    =====

    Section
    -------

    Subsection
    ~~~~~~~~~~

1.2. inline markup
------------------

*italic* \*text\* for emphasis

**boldface** \*\*text\*\* for strong emphasis

``backquote`` \`\`text\`\` for code samples

1.3. lists
----------

1.3.1. bullet list
~~~~~~~~~~~~~~~~~~

- available signs: + - *

  - nested lists must be seperated
  - by an empty line

- and here the parent list continues
  
.. code-block:: rst

  - available signs: + - *

    - nested lists must be seperated
    - by an empty line

  - and here the parent list continues

1.3.2. number list
~~~~~~~~~~~~~~~~~~

#. auto numbering is feasible
#. using prefix sign '#'

.. code-block:: rst

    #. auto numbering is feasible
    #. using prefix sign '#'

1.3.3. definition list
~~~~~~~~~~~~~~~~~~~~~~

term(up to a line)
    Definition must be indented
    and multiple paragraphs are possible

detergent
    a chemical liquid or powder for cleaning

.. code-block:: rst

    term
        Definition must be indented
        and multiple paragraphs are possible

    detergent
        a chemical liquid or powder for cleaning

1.3.4. fields lists
-------------------

:fieldname: Field Content

.. code-block:: rst

    :fieldname: Field Content

    def my_function(my_arg, my_other_arg):
        """A function just for me.

        :param my_arg: The first of my arguments.
        :param my_other_arg: The second of my arguments.

        :returns: A message (just for me, of course).
        """


1.4. line blocks
---------------------

prefix every line with | to preserve line breaks

| These lines are
| broken exactly like in
| the source file

.. code-block:: rst
    
    | These lines are
    | broken exactly like in
    | the source file

1.5. tables
-----------

1.5.1. grid table
~~~~~~~~~~~~~~~~~

For grid tables, grid cells must be painted manually.

+------------------------+------------+----------+----------+
| Header row, column 1   | Header 2   | Header 3 | Header 4 |
| (header rows optional) |            |          |          |
+========================+============+==========+==========+
| body row 1, column 1   | column 2   | column 3 | column 4 |
+------------------------+------------+----------+----------+
| body row 2             | ...        | ...      |          |
+------------------------+------------+----------+----------+

.. code-block:: rst

    +------------------------+------------+----------+----------+
    | Header row, column 1   | Header 2   | Header 3 | Header 4 |
    | (header rows optional) |            |          |          |
    +========================+============+==========+==========+
    | body row 1, column 1   | column 2   | column 3 | column 4 |
    +------------------------+------------+----------+----------+
    | body row 2             | ...        | ...      |          |
    +------------------------+------------+----------+----------+


1.5.2. simple table
~~~~~~~~~~~~~~~~~~~

===  ===  =======
A    B    A and B
===  ===  =======
T    F    F
F    T    F
T    T    T
F    F    F
===  ===  =======

.. code-block:: rst

    ===  ===  =======
    A    B    A and B
    ===  ===  =======
    T    F    F
    F    T    F
    T    T    T
    F    F    F
    ===  ===  =======

1.6. hyper link
---------------

`Link Text <https://example.com>`_

.. code-block:: rst

    `Link Text <https://example.com>`_

`Another Link Text`_

.. _Another Link Text: https://example.com

.. code-block:: rst

    `Another Link Text`_

    .. _Another Link Text: https://example.com

1.7. code blocks
----------------

.. code-block:: rst

    .. code-block:: python

        import sys
        print(sys.executable)

1.8. images
-----------

.. image:: /files/MNIST-with-LeNet/output1.png

.. code-block:: rst

    .. image:: /path/to/image.png

1.9. citation

Lorem ipsum [Ref]_ dolor sit amet.

.. [Ref] Book or article reference, URL or whatever.


2. advanced
===========

2.1. math
---------

According to Leonhard Euler, we have Euler's formula which states that
for any real number x:

    .. math::

        e^ix = cos x + i sin x

.. code-block:: rst

    .. math::

        e^ix = cos x + i sin x

Another famous formula inlined here is :math:`a^2 + b^2 = c^2`.

.. code-block:: rst

    :math:`a^2 + b^2 = c^2`

2.2. emacs rst-mode
-------------------

Since emacs v24.3 reST support is integrated.

2.2.1 section adornment
~~~~~~~~~~~~~~~~~~~~~~~

*C-c C-a C-a*
    rst-adjust


.. code-block:: rst

		My Section Title
		===<C-c C-a C-a>
		
