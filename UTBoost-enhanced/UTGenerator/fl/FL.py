import logging
import re
import pdb
from abc import ABC, abstractmethod

from UTGenerator.augtest.genTest import construct_topn_file_context
from UTGenerator.util.compress_file import get_skeleton
from UTGenerator.util.postprocess_data import extract_code_blocks, extract_locs_for_files
from UTGenerator.util.preprocess_data import (
    get_full_file_paths_and_classes_and_functions,
    get_repo_files,
    line_wrap_content,
    show_project_structure,
)


class FL(ABC):
    def __init__(self, instance_id, structure, problem_statement, test_patch, test_patch_files, **kwargs):
        self.structure = structure
        self.instance_id = instance_id
        self.problem_statement = problem_statement
        self.test_patch = test_patch
        self.test_patch_files = test_patch_files

    @abstractmethod
    def localize(self, top_n=1, mock=False) -> tuple[list, list, list, any]:
        pass


class LLMFL(FL):
    obtain_relevant_files_prompt = """
You are a code reasoning model acting as a test case generator.

Your task is to identify which files are most relevant to the following GitHub problem and need to be tested or modified. Prioritize files that were modified in the original test patch.

---

### GitHub Problem Description ###
{problem_statement}

---

### Original Test Patch ###
{test_patch}

---

### Files modified in the original test patch ###
{test_patch_files}

---

### Repository Structure ###
{structure}

---

### Instructions ###
Return a list of at most 5 full file paths ordered by relevance (most important first). Always include files from the original test patch at the top of the list if relevant.

**Format your response like this, and do not include any explanation:**
file1.py
file2.py

Only include file paths. Wrap the entire list in triple backticks.
"""

#     obtain_relevant_code_prompt = """
# You are a test case generator. Please analyze the following GitHub problem description and the provided repository structure. Your goal is to identify files and locations that need to be modified to add the test cases to see whether the issue can be solved.

# ### GitHub Problem Description ###
# {problem_statement}

# ###

# There is an original test patch that is used to verify the fix.
# ### Original Test Patch ###
# {test_patch}

# ###

# ### File: {file_name} ###
# {file_content}

# ###

# Please provide either the class, the function name or line numbers that need to be edited.
# ### Example 1:
# ```
# class: MyClass
# ```
# ### Example 2:
# ```
# function: my_function
# ```
# ### Example 3:
# ```
# line: 10
# line: 24
# ```

# Return just the location(s)
# """
#     file_content_template = """
# ### File: {file_name} ###
# {file_content}
# """
    file_content_in_block_template = """
### File: {file_name} ###
```python
{file_content}
```
"""
    obtain_relevant_code_combine_top_n_prompt = """
You are a test case generator.

Your goal is to identify **specific code locations** (functions, classes, or lines) in the following files that must be modified or tested in order to address the GitHub problem.

---

### GitHub Problem Description ###
{problem_statement}

---

### Original Test Patch ###
{test_patch}

---

### Relevant Files and Contents ###
{file_contents}

---

### Instructions ###
For each file, return the **exact line numbers, class names, or function names** where changes or tests should be added. Only return necessary items.

**Format your output exactly like this:**
full_path1/file1.py
line: 10
class: MyClass1
function: my_function

full_path2/file2.py
line: 24
function: MyClass2.my_method

Wrap the entire response in triple backticks and include no explanations.
"""

    obtain_relevant_code_combine_top_n_no_line_number_prompt = """
You are a test case generator.

Your task is to identify specific functions or classes in the following files that are relevant to the GitHub issue.

---

### GitHub Problem Description ###
{problem_statement}

---

### Original Test Patch ###
{test_patch}

---

### Relevant Files and Contents ###
{file_contents}

---

### Instructions ###
List only the **function or class names** that are relevant for addressing the issue. Do not include line numbers.

**Format your output like this:**
full_path1/file1.py
function: my_function1
class: MyClass1

full_path2/file2.py
function: MyClass2.my_method
class: MyClass3

Wrap the response in triple backticks. No other text or explanation.
"""
#     obtain_relevant_functions_from_compressed_files_prompt = """
# You are a test case generator. Please look through the following GitHub problem description and the skeleton of relevant files.
# Provide a thorough set of locations that one would need to add test cases to see whether the issue can be solved, including directly related areas as well as any potentially related functions and classes.

# ### GitHub Problem Description ###
# {problem_statement}

# There is an original test patch that is used to verify the fix. You can refer to the file, class, function and locations it modified.
# ### Original Test Patch ###
# {test_patch}

# ###

# ###
# {file_contents}

# ###

# Please provide locations as either the class or the function name.
# ### Examples:
# ```
# full_path1/file1.py
# class: MyClass1

# full_path2/file2.py
# function: MyClass2.my_method

# full_path3/file3.py
# function: my_function
# ```

# Return just the location(s)
# """
    obtain_relevant_functions_and_vars_from_compressed_files_prompt_more = """
You are a code reasoning model acting as a test case generator.

Given a compressed skeleton of code files, identify all **global variables**, **function names**, or **class names** that are relevant to solving the problem.

---

### GitHub Problem Description ###
{problem_statement}

---

### Original Test Patch ###
{test_patch}

---

### Skeleton of Relevant Files ###
{file_contents}

---

### Instructions ###
List only the locations that are likely relevant.

**Format your response exactly like this:**
full_path1/file1.py
function: my_function_1
class: MyClass1

full_path2/file2.py
variable: my_var
function: MyClass2.my_method

Wrap the response in triple backticks. Do not include any explanation.
"""

    def __init__(
        self, instance_id, structure, problem_statement, test_patch, test_patch_files, model_name, backend, **kwargs
    ):
        super().__init__(instance_id, structure, problem_statement, test_patch, test_patch_files)
        self.max_tokens = 300
        self.model_name = model_name
        self.backend = backend

    def _parse_model_return_lines(self, content: str) -> list[str]:
        return content.strip().split("\n")

    def localize(self, top_n=1, mock=False) -> tuple[list, list, list, any]:
        # lazy import, not sure if this is actually better?
        from UTGenerator.util.api_requests import num_tokens_from_messages
        from UTGenerator.util.model import make_model

        found_files = []

        message = self.obtain_relevant_files_prompt.format(
            problem_statement=self.problem_statement,
            test_patch=self.test_patch,
            test_patch_files=self.test_patch_files,
            structure=show_project_structure(self.structure).strip(),
        ).strip()
        # deal with only the test function name
        if 'sensitive' in message:
            message = message.replace('sensitive', 'xxxxxxxxx')
        print(f"prompting with message:\n{message}")
        print("=" * 80)
        if mock:
            traj = {
                "prompt": message,
                "usage": {
                    "prompt_tokens": num_tokens_from_messages(
                        message, "gpt-4o-2024-08-06"
                    ),
                },
            }
            return [], {"raw_output_loc": ""}, traj

        model = make_model(
            model=self.model_name,
            backend=self.backend,
            max_tokens=self.max_tokens,
            temperature=0,
            batch_size=1,
        )
        traj = model.codegen(message, num_samples=1)[0]
        traj["prompt"] = message
        raw_output = traj["response"]
        model_found_files = self._parse_model_return_lines(raw_output)

        files, classes, functions = get_full_file_paths_and_classes_and_functions(
            self.structure
        )

        for file_content in files:
            file = file_content[0]
            if file in model_found_files:
                found_files.append(file)

        # sort based on order of appearance in model_found_files
        found_files = sorted(found_files, key=lambda x: model_found_files.index(x))

        print(raw_output)

        return (
            found_files,
            {"raw_output_files": raw_output},
            traj,
        )

    def localize_function_for_files(
        self, file_names, mock=False
    ) -> tuple[list, dict, dict]:
        from UTGenerator.util.api_requests import num_tokens_from_messages
        from UTGenerator.util.model import make_model

        files, classes, functions = get_full_file_paths_and_classes_and_functions(
            self.structure
        )

        max_num_files = len(file_names)
        while 1:
            # added small fix to prevent too many tokens
            contents = []
            for file_name in file_names[:max_num_files]:
                for file_content in files:
                    if file_content[0] == file_name:
                        content = "\n".join(file_content[1])
                        file_content = line_wrap_content(content)
                        contents.append(
                            self.file_content_template.format(
                                file_name=file_name, file_content=file_content
                            )
                        )
                        break
                else:
                    raise ValueError(f"File {file_name} does not exist.")

            file_contents = "".join(contents)
            if num_tokens_from_messages(file_contents, "gpt-4o-2024-08-06") < 128000:
                break
            else:
                max_num_files -= 1

        message = self.obtain_relevant_code_combine_top_n_prompt.format(
            problem_statement=self.problem_statement,
            test_patch=self.test_patch,
            file_contents=file_contents,
        ).strip()

        # deal with only the test function name
        if 'sensitive' in message:
            message = message.replace('sensitive', 'xxxxxxxxx')

        print(f"prompting with message:\n{message}")
        print("=" * 80)
        if mock:
            traj = {
                "prompt": message,
                "usage": {
                    "prompt_tokens": num_tokens_from_messages(
                        message, "gpt-4o-2024-08-06"
                    ),
                },
            }
            return [], {"raw_output_loc": ""}, traj

        model = make_model(
            model=self.model_name,
            backend=self.backend,
            max_tokens=self.max_tokens,
            temperature=0,
            batch_size=1,
        )
        traj = model.codegen(message, num_samples=1)[0]
        traj["prompt"] = message
        raw_output = traj["response"]

        model_found_locs = extract_code_blocks(raw_output)
        model_found_locs_separated = extract_locs_for_files(
            model_found_locs, file_names
        )

        print(raw_output)

        return model_found_locs_separated, {"raw_output_loc": raw_output}, traj

    def localize_function_from_compressed_files(self, file_names, mock=False):
        from UTGenerator.util.api_requests import num_tokens_from_messages
        from UTGenerator.util.model import make_model

        file_contents = get_repo_files(self.structure, file_names)
        compressed_file_contents = {
            fn: get_skeleton(code) for fn, code in file_contents.items()
        }
        contents = [
            self.file_content_in_block_template.format(file_name=fn, file_content=code)
            for fn, code in compressed_file_contents.items()
        ]
        file_contents = "".join(contents)
        template = (
            self.obtain_relevant_functions_and_vars_from_compressed_files_prompt_more
        )
        message = template.format(
            problem_statement=self.problem_statement, test_patch=self.test_patch, file_contents=file_contents
        )
        
        # deal with only the test function name
        if 'sensitive' in message:
            message = message.replace('sensitive', 'xxxxxxxxx')
            
        
        assert num_tokens_from_messages(message, "gpt-4o-2024-08-06") < 128000
        logging.info(f"prompting with message:\n{message}")
        logging.info("=" * 80)

        if mock:
            traj = {
                "prompt": message,
                "usage": {
                    "prompt_tokens": num_tokens_from_messages(
                        message,
                        self.model_name,
                    ),
                },
            }
            return [], {"raw_output_loc": ""}, traj

        model = make_model(
            model=self.model_name,
            backend=self.backend,
            max_tokens=self.max_tokens,
            temperature=0,
            batch_size=1,
        )
        traj = model.codegen(message, num_samples=1)[0]
        traj["prompt"] = message
        raw_output = traj["response"]
        model_found_locs = extract_code_blocks(raw_output)
        model_found_locs_separated = extract_locs_for_files(
            model_found_locs, file_names
        )

        logging.info(f"==== raw output ====")
        logging.info(raw_output)
        logging.info("=" * 80)
        logging.info(f"==== extracted locs ====")
        for loc in model_found_locs_separated:
            logging.info(loc)
        logging.info("=" * 80)

        print(raw_output)

        return model_found_locs_separated, {"raw_output_loc": raw_output}, traj

    def localize_line_from_coarse_function_locs(
        self,
        file_names,
        coarse_locs,
        context_window: int,
        add_space: bool,
        sticky_scroll: bool,
        no_line_number: bool,
        temperature: float = 0.0,
        num_samples: int = 1,
        mock=False,
    ):
        from UTGenerator.util.api_requests import num_tokens_from_messages
        from UTGenerator.util.model import make_model

        file_contents = get_repo_files(self.structure, file_names)
        topn_content, file_loc_intervals = construct_topn_file_context(
            coarse_locs,
            file_names,
            file_contents,
            self.structure,
            context_window=context_window,
            loc_interval=True,
            add_space=add_space,
            sticky_scroll=sticky_scroll,
            no_line_number=no_line_number,
        )
        if no_line_number:
            template = self.obtain_relevant_code_combine_top_n_no_line_number_prompt
        else:
            template = self.obtain_relevant_code_combine_top_n_prompt
        message = template.format(
            problem_statement=self.problem_statement, test_patch=self.test_patch, file_contents=topn_content
        )
        # deal with only the test function name
        if 'sensitive' in message:
            message = message.replace('sensitive', 'xxxxxxxxx')
        logging.info(f"prompting with message:\n{message}")
        logging.info("=" * 80)
        assert num_tokens_from_messages(message, "gpt-4o-2024-08-06") < 128000
        if mock:
            traj = {
                "prompt": message,
                "usage": {
                    "prompt_tokens": num_tokens_from_messages(
                        message, "gpt-4o-2024-08-06"
                    ),
                },
            }
            return [], {"raw_output_loc": ""}, traj

        model = make_model(
            model=self.model_name,
            backend=self.backend,
            max_tokens=self.max_tokens,
            temperature=temperature,
            batch_size=num_samples,
        )
        raw_trajs = model.codegen(message, num_samples=num_samples)

        # Merge trajectories
        raw_outputs = [raw_traj["response"] for raw_traj in raw_trajs]
        traj = {
            "prompt": message,
            "response": raw_outputs,
            "usage": {  # merge token usage
                "completion_tokens": sum(
                    raw_traj["usage"]["completion_tokens"] for raw_traj in raw_trajs
                ),
                "prompt_tokens": sum(
                    raw_traj["usage"]["prompt_tokens"] for raw_traj in raw_trajs
                ),
            },
        }
        model_found_locs_separated_in_samples = []
        for raw_output in raw_outputs:
            model_found_locs = extract_code_blocks(raw_output)
            model_found_locs_separated = extract_locs_for_files(
                model_found_locs, file_names
            )
            model_found_locs_separated_in_samples.append(model_found_locs_separated)

            logging.info(f"==== raw output ====")
            logging.info(raw_output)
            logging.info("=" * 80)
            print(raw_output)
            print("=" * 80)
            logging.info(f"==== extracted locs ====")
            for loc in model_found_locs_separated:
                logging.info(loc)
            logging.info("=" * 80)
        logging.info("==== Input coarse_locs")
        coarse_info = ""
        for fn, found_locs in coarse_locs.items():
            coarse_info += f"### {fn}\n"
            if isinstance(found_locs, str):
                coarse_info += found_locs + "\n"
            else:
                coarse_info += "\n".join(found_locs) + "\n"
        logging.info("\n" + coarse_info)
        if len(model_found_locs_separated_in_samples) == 1:
            model_found_locs_separated_in_samples = (
                model_found_locs_separated_in_samples[0]
            )

        return (
            model_found_locs_separated_in_samples,
            {"raw_output_loc": raw_outputs},
            traj,
        )
    

# import logging
# import re
# import pdb
# from abc import ABC, abstractmethod

# from UTGenerator.augtest.genTest import construct_topn_file_context
# from UTGenerator.util.compress_file import get_skeleton
# from UTGenerator.util.postprocess_data import extract_code_blocks, extract_locs_for_files
# from UTGenerator.util.preprocess_data import (
#     get_full_file_paths_and_classes_and_functions,
#     get_repo_files,
#     line_wrap_content,
#     show_project_structure,
# )

# class FL(ABC):
#     def __init__(self, instance_id, structure, problem_statement, test_patch, test_patch_files, **kwargs):
#         self.structure = structure
#         self.instance_id = instance_id
#         self.problem_statement = problem_statement
#         self.test_patch = test_patch
#         self.test_patch_files = test_patch_files

#     @abstractmethod
#     def localize(self, top_n=1, mock=False) -> tuple[list, list, list, any]:
#         pass


# class LLMFL(FL):
#     obtain_relevant_files_prompt = """You are a file analysis bot. Your ONLY job is to identify relevant files from the provided context.

# Given the following patch content and a list of files modified by the patch:

# ### Patch Files ###
# {test_patch_files}

# ### Patch Content ###
# {test_patch}

# Based *only* on the information above, list the full paths of the files that should be inspected for adding new tests.
# Prioritize the files listed in "Patch Files".

# ### Output Format:
# Return only the file paths inside a markdown code block. Do not add any explanation.
# Example:
# testing/logging/test_reporting.py
# src/_pytest/logging.py

# """

# #     obtain_relevant_code_prompt = """
# # You are a test case generator. Please analyze the following GitHub problem description and the provided repository structure. Your goal is to identify files and locations that need to be modified to add the test cases to see whether the issue can be solved.

# # ### GitHub Problem Description ###
# # {problem_statement}

# # ###

# # There is an original test patch that is used to verify the fix.
# # ### Original Test Patch ###
# # {test_patch}

# # ###

# # ### File: {file_name} ###
# # {file_content}

# # ###

# # Please provide either the class, the function name or line numbers that need to be edited.
# # ### Example 1:
# # ```
# # class: MyClass
# # ```
# # ### Example 2:
# # ```
# # function: my_function
# # ```
# # ### Example 3:
# # ```
# # line: 10
# # line: 24
# # ```

# # Return just the location(s)
# # """
# #     file_content_template = """
# # ### File: {file_name} ###
# # {file_content}
# # """

#     obtain_relevant_code_combine_top_n_prompt = """You are a code analysis bot. Your job is to extract function and class names from code that are mentioned in a patch.

# You are given:
# - An `Original Patch` that shows code changes.
# - The full content of `Code Files` that were changed.

# Your task is to:
# 1. Look at the `Original Patch` to see which functions or classes were modified (check the `@@ ... @@` lines).
# 2. Find those same functions or classes in the full `Code Files`.
# 3. List the file path and the function/class names you found.

# ---
# ### Original Patch ###
# {test_patch}
# ---
# ### Code Files ###
# {file_contents}
# ---
# ### Output Format:
# List the locations for each file using this exact format:

# full_path1/file1.py
# function: function_name_from_patch
# class: TestClassNameFromPatch

# Only return the code block. Do not add explanations.
# """
    
#     obtain_relevant_functions_and_vars_from_compressed_files_prompt_more = """You are a code analysis bot. Your job is to extract function and class names from a code patch.

# You are given:
# - An `Original Patch` that shows code changes.
# - Skeletons of the files that were changed.

# Your task is to analyze the `Original Patch` and list the names of the functions and classes that were modified.

# ---
# ### Original Patch ###
# {test_patch}
# ---
# ### Skeleton of Relevant Files ###
# {file_contents}
# ---
# ### Output Format:
# List the file path and the function/class names you found in the patch.
# ### Examples:
# full_path1/file1.py
# function: my_function_1
# class: MyClass1

# If nothing relevant exists, return an empty response. Do not explain your choices.
# """

#     # This prompt is not used in the primary flow but is also simplified.
#     obtain_relevant_code_combine_top_n_no_line_number_prompt = obtain_relevant_code_combine_top_n_prompt

#     file_content_in_block_template = """
# ### File: {file_name} ###
# ```python
# {file_content}
# """
# #     obtain_relevant_functions_from_compressed_files_prompt = """
# # You are a test case generator. Please look through the following GitHub problem description and the skeleton of relevant files.
# # Provide a thorough set of locations that one would need to add test cases to see whether the issue can be solved, including directly related areas as well as any potentially related functions and classes.

# # ### GitHub Problem Description ###
# # {problem_statement}

# # There is an original test patch that is used to verify the fix. You can refer to the file, class, function and locations it modified.
# # ### Original Test Patch ###
# # {test_patch}

# # ###

# # ###
# # {file_contents}

# # ###

# # Please provide locations as either the class or the function name.
# # ### Examples:
# # ```
# # full_path1/file1.py
# # class: MyClass1

# # full_path2/file2.py
# # function: MyClass2.my_method

# # full_path3/file3.py
# # function: my_function
# # ```

# # Return just the location(s)
# # """
#     def __init__(
#         self, instance_id, structure, problem_statement, test_patch, test_patch_files, model_name, backend, **kwargs
#     ):
#         super().__init__(instance_id, structure, problem_statement, test_patch, test_patch_files)
#         self.max_tokens = 300
#         self.model_name = model_name
#         self.backend = backend

#     # def _parse_model_return_lines(self, content: str) -> list[str]:
#     #     return content.strip().split("\n")

#     def _parse_model_return_lines(self, content: str) -> list[str]:
#         if not content:
#             return []
#         if content.startswith("```") and content.endswith("```"):
#             content = content.strip("```").strip()
#             lines = content.split('\n')
#             if lines and lines[0].lower() in ['python', 'py', 'text', 'sh', 'markdown', '']:
#                 content = '\n'.join(lines[1:])
#         return [line for line in content.strip().split("\n") if line.strip()]
    

#     def localize(self, top_n=1, mock=False) -> tuple[list, list, list, any]:
#         from UTGenerator.util.api_requests import create_chatgpt_config, request_chatgpt_engine, num_tokens_from_messages
#         from UTGenerator.util.preprocess_data import show_project_structure

#         message = self.obtain_relevant_files_prompt.format(
#             problem_statement=self.problem_statement,
#             test_patch=self.test_patch,
#             test_patch_files=self.test_patch_files,
#             structure=show_project_structure(self.structure).strip(),
#         ).strip()

#         if 'sensitive' in message:
#             message = message.replace('sensitive', 'xxxxxxxxx')

#         if mock:
#             traj = {
#                 "prompt": message,
#                 "usage": {
#                     "prompt_tokens": num_tokens_from_messages(message, self.model_name),
#                 },
#             }
#             return [], {"raw_output_loc": ""}, traj

#         cfg = create_chatgpt_config(
#             message=message,
#             model=self.model_name,
#             # tools=tools,      <-- Step 2: Comment out or delete this line
#             temperature=0,
#             max_tokens=self.max_tokens,
#         )

#         outputs, usage = request_chatgpt_engine(cfg)

#         raw_output = outputs[0] if outputs else ""
#         print(raw_output)

#         model_found_files = self._parse_model_return_lines(raw_output)

#         files, classes, functions = get_full_file_paths_and_classes_and_functions(self.structure)
#         file_set = {f[0] for f in files}
#         found_files = [f for f in model_found_files if f in file_set]
        
#         found_files = sorted(found_files, key=lambda x: model_found_files.index(x))

#         traj = {
#             "prompt": message,
#             "response": raw_output,
#             "usage": usage,
#         }

#         return found_files, {"raw_output_files": raw_output}, traj


#     def localize_function_for_files(
#         self, file_names, mock=False
#     ) -> tuple[list, dict, dict]:
#         from UTGenerator.util.api_requests import num_tokens_from_messages
#         from UTGenerator.util.model import make_model

#         files, classes, functions = get_full_file_paths_and_classes_and_functions(
#             self.structure
#         )

#         max_num_files = len(file_names)
#         while 1:
#             # added small fix to prevent too many tokens
#             contents = []
#             for file_name in file_names[:max_num_files]:
#                 for file_content in files:
#                     if file_content[0] == file_name:
#                         content = "\n".join(file_content[1])
#                         file_content = line_wrap_content(content)
#                         contents.append(
#                             self.file_content_template.format(
#                                 file_name=file_name, file_content=file_content
#                             )
#                         )
#                         break
#                 else:
#                     raise ValueError(f"File {file_name} does not exist.")

#             file_contents = "".join(contents)
#             if num_tokens_from_messages(file_contents, self.model_name) < 128000:
#                 break
#             else:
#                 max_num_files -= 1

#         message = self.obtain_relevant_code_combine_top_n_prompt.format(
#             problem_statement=self.problem_statement,
#             test_patch=self.test_patch,
#             file_contents=file_contents,
#         ).strip()

#         # deal with only the test function name
#         if 'sensitive' in message:
#             message = message.replace('sensitive', 'xxxxxxxxx')

#         print(f"prompting with message:\n{message}")
#         print("=" * 80)
#         if mock:
#             traj = {
#                 "prompt": message,
#                 "usage": {
#                     "prompt_tokens": num_tokens_from_messages(
#                         message, self.model_name
#                     ),
#                 },
#             }
#             return [], {"raw_output_loc": ""}, traj

#         model = make_model(
#             model=self.model_name,
#             backend=self.backend,
#             max_tokens=self.max_tokens,
#             temperature=0,
#             batch_size=1,
#         )
#         traj = model.codegen(message, num_samples=1)[0]
#         traj["prompt"] = message
#         raw_output = traj["response"]

#         model_found_locs = extract_code_blocks(raw_output)
#         model_found_locs_separated = extract_locs_for_files(
#             model_found_locs, file_names
#         )

#         print(raw_output)

#         return model_found_locs_separated, {"raw_output_loc": raw_output}, traj

#     def localize_function_from_compressed_files(self, file_names, mock=False):
#         from UTGenerator.util.api_requests import num_tokens_from_messages
#         from UTGenerator.util.model import make_model

#         file_contents = get_repo_files(self.structure, file_names)
#         compressed_file_contents = {
#             fn: get_skeleton(code) for fn, code in file_contents.items()
#         }
#         contents = [
#             self.file_content_in_block_template.format(file_name=fn, file_content=code)
#             for fn, code in compressed_file_contents.items()
#         ]
#         file_contents = "".join(contents)
#         template = (
#             self.obtain_relevant_functions_and_vars_from_compressed_files_prompt_more
#         )
#         message = template.format(
#             problem_statement=self.problem_statement, test_patch=self.test_patch, file_contents=file_contents
#         )
        
#         # deal with only the test function name
#         if 'sensitive' in message:
#             message = message.replace('sensitive', 'xxxxxxxxx')
            
#         assert num_tokens_from_messages(message, self.model_name) < 128000
#         logging.info(f"prompting with message:\n{message}")
#         logging.info("=" * 80)

#         if mock:
#             traj = {
#                 "prompt": message,
#                 "usage": {
#                     "prompt_tokens": num_tokens_from_messages(
#                         message, self.model_name
#                     ),
#                 },
#             }
#             return [], {"raw_output_loc": ""}, traj

#         model = make_model(
#             model=self.model_name,
#             backend=self.backend,
#             max_tokens=self.max_tokens,
#             temperature=0,
#             batch_size=1,
#         )
#         traj = model.codegen(message, num_samples=1)[0]
#         traj["prompt"] = message
#         raw_output = traj["response"]
#         model_found_locs = extract_code_blocks(raw_output)
#         model_found_locs_separated = extract_locs_for_files(
#             model_found_locs, file_names
#         )

#         logging.info(f"==== raw output ====")
#         logging.info(raw_output)
#         logging.info("=" * 80)
#         logging.info(f"==== extracted locs ====")
#         for loc in model_found_locs_separated:
#             logging.info(loc)
#         logging.info("=" * 80)

#         print(raw_output)

#         return model_found_locs_separated, {"raw_output_loc": raw_output}, traj

#     def localize_line_from_coarse_function_locs(
#         self,
#         file_names,
#         coarse_locs,
#         context_window: int,
#         add_space: bool,
#         sticky_scroll: bool,
#         no_line_number: bool,
#         temperature: float = 0.0,
#         num_samples: int = 1,
#         mock=False,
#     ):
#         from UTGenerator.util.api_requests import num_tokens_from_messages
#         from UTGenerator.util.model import make_model

#         file_contents = get_repo_files(self.structure, file_names)
#         topn_content, file_loc_intervals = construct_topn_file_context(
#             coarse_locs,
#             file_names,
#             file_contents,
#             self.structure,
#             context_window=context_window,
#             loc_interval=True,
#             add_space=add_space,
#             sticky_scroll=sticky_scroll,
#             no_line_number=no_line_number,
#         )
#         if no_line_number:
#             template = self.obtain_relevant_code_combine_top_n_no_line_number_prompt
#         else:
#             template = self.obtain_relevant_code_combine_top_n_prompt
#         message = template.format(
#             problem_statement=self.problem_statement, test_patch=self.test_patch, file_contents=topn_content
#         )
#         # deal with only the test function name
#         if 'sensitive' in message:
#             message = message.replace('sensitive', 'xxxxxxxxx')
#         logging.info(f"prompting with message:\n{message}")
#         logging.info("=" * 80)
#         assert num_tokens_from_messages(message, self.model_name) < 128000
#         if mock:
#             traj = {
#                 "prompt": message,
#                 "usage": {
#                     "prompt_tokens": num_tokens_from_messages(
#                         message, self.model_name
#                     ),
#                 },
#             }
#             return [], {"raw_output_loc": ""}, traj

#         model = make_model(
#             model=self.model_name,
#             backend=self.backend,
#             max_tokens=self.max_tokens,
#             temperature=temperature,
#             batch_size=num_samples,
#         )
#         raw_trajs = model.codegen(message, num_samples=num_samples)

#         # Merge trajectories
#         raw_outputs = [raw_traj["response"] for raw_traj in raw_trajs]
#         traj = {
#             "prompt": message,
#             "response": raw_outputs,
#             "usage": {  # merge token usage
#                 "completion_tokens": sum(
#                     raw_traj["usage"]["completion_tokens"] for raw_traj in raw_trajs
#                 ),
#                 "prompt_tokens": sum(
#                     raw_traj["usage"]["prompt_tokens"] for raw_traj in raw_trajs
#                 ),
#             },
#         }
#         model_found_locs_separated_in_samples = []
#         for raw_output in raw_outputs:
#             model_found_locs = extract_code_blocks(raw_output)
#             model_found_locs_separated = extract_locs_for_files(
#                 model_found_locs, file_names
#             )
#             model_found_locs_separated_in_samples.append(model_found_locs_separated)

#             logging.info(f"==== raw output ====")
#             logging.info(raw_output)
#             logging.info("=" * 80)
#             print(raw_output)
#             print("=" * 80)
#             logging.info(f"==== extracted locs ====")
#             for loc in model_found_locs_separated:
#                 logging.info(loc)
#             logging.info("=" * 80)
#         logging.info("==== Input coarse_locs")
#         coarse_info = ""
#         for fn, found_locs in coarse_locs.items():
#             coarse_info += f"### {fn}\n"
#             if isinstance(found_locs, str):
#                 coarse_info += found_locs + "\n"
#             else:
#                 coarse_info += "\n".join(found_locs) + "\n"
#         logging.info("\n" + coarse_info)
#         if len(model_found_locs_separated_in_samples) == 1:
#             model_found_locs_separated_in_samples = (
#                 model_found_locs_separated_in_samples[0]
#             )

#         return (
#             model_found_locs_separated_in_samples,
#             {"raw_output_loc": raw_outputs},
#             traj,
#         )