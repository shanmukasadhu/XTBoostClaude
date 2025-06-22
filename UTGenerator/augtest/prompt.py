edit_relevant_file_instruction = """
Below are some code segments, each from a relevant file. One or more of these files may be edited to add the test cases.
Please generate the test cases to see whether the issue can be solved, do not remember to import the functions before you use them.
"""
edit_relevant_file_with_scope_instruction = """
Below are some code segments, each from a relevant file. One or more of these files may be edited to add the test cases.
In the file below, "..." refers to some less relevant content being omited for brebity.
"""
with_scope_explanation = """
Note that "..." refers to some omited content that is not actually in the files. Your *SEARCH/REPLACE* edit must not contain such "...".
"""
edit_relevant_file_with_suspicious_loc_instruction = """
Below are some code segments, each from a relevant file. One or more of these files may be edited to add the test cases. Some suspicious locations are provided for closer inspection.
"""
edit_prompt_combine_topn = """
We are currently generating test cases for the following issue within our repository. Here is the issue text:
--- BEGIN ISSUE ---
{problem_statement}
--- END ISSUE ---

There is an original test patch that is used to verify the fix. You can learn from it to generate new test cases that thoroughly test the fix.
--- Original Test Patch ---
```
{test_patch}
```
--- END Original Test Patch ---

{edit_relevant_file_instruction}
--- BEGIN FILE ---
```
{content}
```
--- END FILE ---

Please generate `edit_file` commands to fix the issue.

The `edit_file` command takes four arguments:

edit_file(filename: str, start: int, end: int, content: str) -> None:
    Edit a file. It replaces lines `start` through `end` (inclusive) with the given text `content` in the open file.
    Args:
    filename: str: The full file name to edit.
    start: int: The start line number. Must satisfy start >= 1.
    end: int: The end line number. Must satisfy start <= end <= number of lines in the file.
    content: str: The content to replace the lines with.

Please note that THE `edit_file` FUNCTION REQUIRES PROPER INDENTATION. If you would like to add the line '        print(x)', you must fully write that out, with all those spaces before the code!
Wrap the `edit_file` command in blocks ```python...```.
"""

find_diffrence_between_two_code_blocks = """
Here are two code blocks. Please find the difference between them. We need to generate the test cases to see whether the issue can be solved. You should return the information of the difference, so it can be used to generate the test cases.
--- BEGIN CODE BLOCK 1 ---
{code_block_1}
--- END CODE BLOCK 1 ---

--- BEGIN CODE BLOCK 2 ---
{code_block_2}
--- END CODE BLOCK 2 ---
"""

edit_prompt_combine_topn_cot = """
We are currently generating test cases for the following issue within our repository. Here is the issue text:
--- BEGIN ISSUE ---
{problem_statement}
--- END ISSUE ---

There is an original test patch that is used to verify the fix. You can learn from it to generate new test cases that thoroughly test the fix.
--- Original Test Patch ---
```
{test_patch}
```
--- END Original Test Patch ---

{edit_relevant_file_instruction}
--- BEGIN FILE ---
```
{content}
```
--- END FILE ---

Please first generate the test cases to see whether the issue can be solved.

The `edit_file` command takes four arguments:

edit_file(filename: str, start: int, end: int, content: str) -> None:
    Edit a file. It replaces lines `start` through `end` (inclusive) with the given text `content` in the open file.
    Args:
    filename: str: The full file name to edit.
    start: int: The start line number. Must satisfy start >= 1.
    end: int: The end line number. Must satisfy start <= end <= number of lines in the file.
    content: str: The content to replace the lines with.

Please note that THE `edit_file` FUNCTION REQUIRES PROPER INDENTATION. If you would like to add the line '        print(x)', you must fully write that out, with all those spaces before the code!
Wrap the `edit_file` command in blocks ```python...```.
"""

# This is the active 
edit_prompt_combine_topn_cot_diff = """
We are currently adding test cases for verifying whether the following issue is resolved. Here is the issue text:
--- BEGIN ISSUE ---
{problem_statement}
--- END ISSUE ---

There is an original test patch that is used to verify the fix. 
--- Original Test Patch ---
```
{test_patch}
```
--- END Original Test Patch ---
Here are some guidelines:
1. You should first analyze whether the original test patch conforms to the issue. If the original test patch cannot thoroughly test the fix, generate new test cases to cover more aspects of the fix.
2. Feel free to add more test functions to thoroughly test the fix. Do not remove original test cases.
3. If any required import for the test cases is missing, add it to the file. If it's already present, do not duplicate the import. Make sure to import any necessary packages, modules, or functions before using them in the test cases.
4. You should be aware that some issues describe the incorrect behavior of the code, while some describe the correct behavior of the code. If the issue describes the incorrect behavior, please generate the test cases that expose the issue. If the issue describes the correct behavior, please generate the test cases that the correct behavior can be verified.
5. For assertion failures, you should generate the test cases that can reproduce the assertion failure.

There is a list of imports that are used in the original test patch. You should import them in the test case if you want to use some functions in the original test patch. Or if you want to add new functions, you should also import the functions you need.
--- Added Imports ---
```
{added_imports}
```
--- END Added Imports ---

{edit_relevant_file_instruction}
--- BEGIN FILE ---
```
{content}
```
--- END FILE ---

Your mission is to generate the test cases to see whether the issue can be solved, and check the related original function are not affected by the fix. Try to generate the cases to thoroughly test the fix. Do not remove the original function (we add test cases, but not subtract test cases or function).
Please first localize the places to add the cases along with import functions based on the issue statement, and then generate *SEARCH/REPLACE* edits to add the test cases. You should generate *SEARCH/REPLACE* edits for everywhere you want to edit. 

Every *SEARCH/REPLACE* edit must use this format:
1. The file path
2. The start of search block: <<<<<<< SEARCH
3. A contiguous chunk of lines to search for in the existing source code
4. The dividing line: =======
5. The lines to replace into the source code
6. The end of the replace block: >>>>>>> REPLACE

Here is an example:

```python
### mathweb/flask/app.py
<<<<<<< SEARCH
from flask import Flask
=======
import math
from flask import Flask
>>>>>>> REPLACE
```

Please note that the *SEARCH/REPLACE* edit REQUIRES PROPER INDENTATION. If you would like to add the line '        print(x)', you must fully write that out, with all those spaces before the code!
Wrap the *SEARCH/REPLACE* edit in blocks ```python...```.
"""