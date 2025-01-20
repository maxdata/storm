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
        """Initialize the article generation module.
        
        Args:
            engine (Union[dspy.dsp.LM, dspy.dsp.HFModel]): The language model engine 
                used for text generation and processing.
        """
        super().__init__()  # Initialize the base class
        self.write_section = dspy.Predict(WriteSection)  # Prediction model for writing sections
        self.engine = engine  # Store the engine for later use

    def _get_cited_information_string(
        self,
        all_citation_index: Set[int],  # Set of citation indices
        knowledge_base: KnowledgeBase,  # Knowledge base containing information
        max_words: int = 1500,  # Maximum number of words to include
    ):
        """
        Generate formatted information text from citation indices and knowledge base.

        Args:
            all_citation_index (Set[int]): Set of citation indices to process
            knowledge_base (KnowledgeBase): Knowledge base containing the information
            max_words (int, optional): Maximum number of words to include. Defaults to 1500.

        Returns:
            str: Formatted string containing information from citations, joined by newlines
        """
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

    def synthesize_node(
        self,
        node,
        topic: str,
        knowledge_base,
    ):
        """
        Synthesizes content for a given node by either returning existing output or generating new content.
        
        Args:
            node: The node object containing content to synthesize
            topic (str): The main topic for content generation
            knowledge_base: The knowledge base containing citation iznformation
            
        Returns:
            str: The synthesized content for the node
        """
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

    def forward(self, knowledge_base: KnowledgeBase):
        """
        Process a knowledge base to generate a structured document with paragraphs for each node.
        
        Args:
            knowledge_base (KnowledgeBase): The knowledge base containing nodes to process
            
        Returns:
            dict: A mapping of node paths to generated paragraphs
        """
        all_nodes = knowledge_base.collect_all_nodes()
        node_to_paragraph = {}

        def _node_generate_paragraph(node):
            """
            Generate a paragraph for a single node in the knowledge base.
            
            Args:
                node: The node to generate content for
                
            Returns:
                tuple: A tuple containing (node_path, generated_paragraph)
            """
            node_gen_paragraph = self.synthesize_node(
                node=node, topic=knowledge_base.topic, knowledge_base=knowledge_base
            )
            lines = node_gen_paragraph.split("\n")
            if lines[0].strip().replace("*", "").replace("#", "") == node.name:
                lines = lines[1:]
            node_gen_paragraph = "\n".join(lines)
            path = " -> ".join(node.get_path_from_root())
            return path, node_gen_paragraph

        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_node = {
                executor.submit(_node_generate_paragraph, node): node
                for node in all_nodes
            }

            for future in as_completed(future_to_node):
                path, node_gen_paragraph = future.result()
                node_to_paragraph[path] = node_gen_paragraph

        def helper(cur_root, level):
            """
            Recursively build the document structure from the knowledge base nodes.
            
            Args:
                cur_root: The current node being processed
                level (int): The current heading level in the document
                
            Returns:
                list: List of formatted document sections
            """
            to_return = []
            if cur_root is not None:
                hash_tag = "#" * level + " "
                cur_path = " -> ".join(cur_root.get_path_from_root())
                node_gen_paragraph = node_to_paragraph[cur_path]
                to_return.append(f"{hash_tag}{cur_root.name}\n{node_gen_paragraph}")

            for child in cur_root.children:
                to_return.extend(helper(child, level + 1))
            return to_return

        to_return = []
        for child in knowledge_base.root.children:
            to_return.extend(helper(child, level=1))

        return "\n".join(to_return)


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
