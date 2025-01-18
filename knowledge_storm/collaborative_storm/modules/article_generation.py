import dspy  # Importing the dspy library for handling DSP modules and operations
from concurrent.futures import ThreadPoolExecutor, as_completed  # For parallel execution
from typing import Set, Union  # For type hinting

from .collaborative_storm_utils import clean_up_section  # Utility function for cleaning up sections
from ...dataclass import KnowledgeBase, KnowledgeNode  # Data classes for knowledge representation


class ArticleGenerationModule(dspy.Module):
    """Use the information collected from the information-seeking conversation to write a section."""

    def __init__(
        self,
        engine: Union[dspy.dsp.LM, dspy.dsp.HFModel],  # Engine for language model processing
    ):
        super().__init__()  # Initialize the base class
        self.write_section = dspy.Predict(WriteSection)  # Prediction model for writing sections
        self.engine = engine  # Store the engine for later use

    def _get_cited_information_string(
        self,
        all_citation_index: Set[int],  # Set of citation indices
        knowledge_base: KnowledgeBase,  # Knowledge base containing information
        max_words: int = 1500,  # Maximum number of words to include
    ):
        information = []  # List to store information strings
        cur_word_count = 0  # Current word count
        for index in sorted(list(all_citation_index)):  # Iterate over sorted citation indices
            info = knowledge_base.info_uuid_to_info_dict[index]  # Retrieve information by index
            snippet = info.snippets[0]  # Get the first snippet of information
            info_text = f"[{index}]: {snippet} (Question: {info.meta['question']}. Query: {info.meta['query']})"  # Format the information text
            cur_snippet_length = len(info_text.split())  # Calculate the length of the snippet
            if cur_snippet_length + cur_word_count > max_words:  # Check if adding this snippet exceeds max words
                break  # Stop if max words exceeded
            cur_word_count += cur_snippet_length  # Update current word count
            information.append(info_text)  # Add formatted text to information list
        return "\n".join(information)  # Return the information as a single string

    def gen_section(
        self, topic: str, node: KnowledgeNode, knowledge_base: KnowledgeBase  # Generate a section for a given topic and node
    ):
        if node is None or len(node.content) == 0:  # Check if node is empty
            return ""  # Return empty string if no content
        if (
            node.synthesize_output is not None
            and node.synthesize_output
            and not node.need_regenerate_synthesize_output  # Check if synthesis output is already generated and valid
        ):
            return node.synthesize_output  # Return existing synthesis output
        all_citation_index = node.collect_all_content()  # Collect all citation indices
        information = self._get_cited_information_string(
            all_citation_index=all_citation_index, knowledge_base=knowledge_base  # Get cited information string
        )
        with dspy.settings.context(lm=self.engine):  # Set the language model context
            synthesize_output = clean_up_section(
                self.write_section(
                    topic=topic, info=information, section=node.name  # Generate section using the write_section model
                ).output
            )
        node.synthesize_output = synthesize_output  # Store the synthesized output in the node
        node.need_regenerate_synthesize_output = False  # Mark that regeneration is not needed
        return node.synthesize_output  # Return the synthesized output

    def forward(self, knowledge_base: KnowledgeBase):  # Forward method to process the knowledge base
        all_nodes = knowledge_base.collect_all_nodes()  # Collect all nodes from the knowledge base
        node_to_paragraph = {}  # Dictionary to map node paths to paragraphs

        # Define a function to generate paragraphs for nodes
        def _node_generate_paragraph(node):
            node_gen_paragraph = self.gen_section(
                topic=knowledge_base.topic, node=node, knowledge_base=knowledge_base  # Generate section for the node
            )
            lines = node_gen_paragraph.split("\n")  # Split the generated paragraph into lines
            if lines[0].strip().replace("*", "").replace("#", "") == node.name:  # Check if the first line is the node name
                lines = lines[1:]  # Remove the first line if it is the node name
            node_gen_paragraph = "\n".join(lines)  # Join the lines back into a paragraph
            path = " -> ".join(node.get_path_from_root())  # Get the path from root to the node
            return path, node_gen_paragraph  # Return the path and paragraph

        with ThreadPoolExecutor(max_workers=5) as executor:  # Use a thread pool executor for parallel processing
            # Submit all tasks
            future_to_node = {
                executor.submit(_node_generate_paragraph, node): node  # Submit node generation tasks
                for node in all_nodes
            }

            # Collect the results as they complete
            for future in as_completed(future_to_node):  # Iterate over completed futures
                path, node_gen_paragraph = future.result()  # Get the result of the future
                node_to_paragraph[path] = node_gen_paragraph  # Map the path to the generated paragraph

        def helper(cur_root, level):  # Helper function to recursively build the document
            to_return = []  # List to store the document parts
            if cur_root is not None:  # Check if the current root is not None
                hash_tag = "#" * level + " "  # Create a hash tag for the current level
                cur_path = " -> ".join(cur_root.get_path_from_root())  # Get the path from root to the current node
                node_gen_paragraph = node_to_paragraph[cur_path]  # Get the generated paragraph for the current path
                to_return.append(f"{hash_tag}{cur_root.name}\n{node_gen_paragraph}")  # Append the formatted section
                for child in cur_root.children:  # Iterate over the children of the current root
                    to_return.extend(helper(child, level + 1))  # Recursively process each child
            return to_return  # Return the document parts

        to_return = []  # List to store the final document
        for child in knowledge_base.root.children:  # Iterate over the children of the root node
            to_return.extend(helper(child, level=1))  # Build the document starting from each child

        return "\n".join(to_return)  # Return the final document as a single string


class WriteSection(dspy.Signature):
    """Write a Wikipedia section based on the collected information. You will be given the topic, the section you are writing and relevant information.
    Each information will be provided with the raw content along with question and query lead to that information.
    Here is the format of your writing:
    Use [1], [2], ..., [n] in line (for example, "The capital of the United States is Washington, D.C.[1][3]."). You DO NOT need to include a References or Sources section to list the sources at the end.
    """

    info = dspy.InputField(prefix="The collected information:\n", format=str)  # Input field for collected information
    topic = dspy.InputField(prefix="The topic of the page: ", format=str)  # Input field for the topic
    section = dspy.InputField(prefix="The section you need to write: ", format=str)  # Input field for the section name
    output = dspy.OutputField(
        prefix="Write the section with proper inline citations (Start your writing. Don't include the page title, section name, or try to write other sections. Do not start the section with topic name.):\n",
        format=str,  # Output field for the generated section
    )
