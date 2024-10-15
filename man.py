from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.llms import OpenAI
from pydantic import BaseModel, Field
from typing import List, Literal
import os
from manim import *
import numpy as np
import textwrap
import re
import json

os.environ["OPENAI_API_KEY"] = "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"

def preprocess_json(json_string):
    json_string = re.sub(r',\s*([}\]])', r'\1', json_string)
    
    open_quotes = 0
    open_braces = 0
    open_brackets = 0
    fixed_json = []
    
    for char in json_string:
        if char == '"' and fixed_json[-1] != '\\':
            open_quotes = 1 - open_quotes
        elif open_quotes == 0:
            if char == '{':
                open_braces += 1
            elif char == '}':
                open_braces -= 1
            elif char == '[':
                open_brackets += 1
            elif char == ']':
                open_brackets -= 1
        
        fixed_json.append(char)
    
    fixed_json.extend(['"'] * open_quotes)
    fixed_json.extend(['}'] * open_braces)
    fixed_json.extend([']'] * open_brackets)
    
    return ''.join(fixed_json)

class Topic(BaseModel):
    title: str = Field(description="The title of the topic")
    animation_type: Literal["unit_circle", "graph", "formula", "text"] = Field(description="The type of animation for this topic")
    content: str = Field(description="The content for this topic. For 'graph', this should be only a valid function to graph, not a formula but actual examples. For 'formula', this should be only the formula.")

class Curriculum(BaseModel):
    topics: List[Topic] = Field(description="List of topics in the curriculum")

class AIVideoEducationVisualizer(Scene):
    def __init__(self, curriculum_data, main_topic):
        super().__init__()
        self.curriculum_data = curriculum_data
        self.main_topic = main_topic

    def construct(self):
        print("Starting construct method")
        title = Text(f"{self.main_topic.capitalize()} Lesson", font_size=48)
        self.play(Write(title))
        self.wait(2)
        self.play(FadeOut(title))

        topics = self.curriculum_data.topics
        if not topics:
            self.create_error_message("No topics found in the curriculum.")
            return

        print(f"Number of topics: {len(topics)}")
        for index, topic in enumerate(topics):
            print(f"Processing topic {index + 1}/{len(topics)}")
            if isinstance(topic, Topic):
                self.create_topic_animation(topic)
            else:
                print(f"Error: Invalid topic data at index {index}. Expected Topic object, got {type(topic)}")
                self.create_error_message(f"Invalid topic data at index {index}")

    def create_topic_animation(self, topic):
        if not isinstance(topic, Topic):
            print(f"Error: Invalid topic data. Expected Topic object, got {type(topic)}")
            self.create_error_message("Invalid topic data")
            return

        print(f"Processing topic: {topic.title}")
        title = Text(topic.title, font_size=36)
        self.play(Write(title))
        self.wait(1)
        self.play(title.animate.to_edge(UP))

        print(f"Animation type: {topic.animation_type}")
        print(f"Content: {topic.content}")

        if topic.animation_type == 'unit_circle':
            self.create_unit_circle_animation()
        elif topic.animation_type == 'graph':
            self.create_graph_animation(topic.content)
        elif topic.animation_type == 'formula':
            self.create_formula_animation(topic.content)
        elif topic.animation_type == 'text':
            self.create_text_animation(topic.content)
        else:
            print(f"Unknown animation type: {topic.animation_type}")
            self.create_error_message(f"Unknown animation type: {topic.animation_type}")

        self.wait(2)
        self.play(FadeOut(title))

    def create_text_animation(self, content):
        print("Creating text animation")
        wrapped_text = textwrap.fill(content, width=50)
        content_text = Text(wrapped_text, font_size=24)
        content_text.move_to(ORIGIN)
        self.play(Write(content_text))
        self.wait(2)
        self.play(FadeOut(content_text))

    def create_unit_circle_animation(self):
        print("Creating unit circle animation")
        circle = Circle(radius=2, color=BLUE)
        self.play(Create(circle))
        
        angle = ValueTracker(0)
        line = Line(ORIGIN, 2*RIGHT, color=RED)
        dot = Dot(color=YELLOW)
        
        line.add_updater(lambda m: m.set_angle(angle.get_value()))
        dot.add_updater(lambda m: m.move_to(line.get_end()))
        
        self.add(line, dot)
        self.play(angle.animate.set_value(2*PI), run_time=4)
        self.wait()
        self.play(FadeOut(circle), FadeOut(line), FadeOut(dot))

    def create_graph_animation(self, function_str):
        print(f"Creating graph animation for function: {function_str}")
        ax = Axes(
            x_range=[-5, 5, 1],
            y_range=[-5, 5, 1],
            axis_config={"color": BLUE},
        )
        
        match = re.search(r'y\s*=\s*(.+)', function_str)
        if match:
            func_str = match.group(1).lower()
            func_str = func_str.replace('^', '**') 
            
            try:
                def safe_eval(x):
                    return eval(func_str, {"__builtins__": None}, {"x": x, "abs": abs, "sin": np.sin, "cos": np.cos, "tan": np.tan, "sqrt": np.sqrt, "log": np.log, "e": np.e})
                
                graph = ax.plot(safe_eval, color=RED)
                label = MathTex(match.group(0), font_size=24)
                
                label.next_to(ax, UP)
                self.play(Create(ax), Create(graph), Write(label))
                self.wait(2)
                self.play(FadeOut(ax), FadeOut(graph), FadeOut(label))
            except Exception as e:
                print(f"Error creating graph for function {func_str}: {e}")
                self.create_error_message(f"Unable to graph: {match.group(0)}")
        else:
            self.create_error_message("No valid function found for graphing")

    def create_formula_animation(self, formula):
        print(f"Creating formula animation: {formula}")
        try:
            math_formula = MathTex(formula)
            self.play(Write(math_formula))
            self.wait(2)
            self.play(FadeOut(math_formula))
        except Exception as e:
            print(f"Error creating LaTeX formula: {e}")
            self.create_text_animation(f"Formula: {formula}")

    def create_error_message(self, message):
        print(f"Creating error message: {message}")
        error_text = Text(message, color=RED)
        self.play(Write(error_text))
        self.wait(2)
        self.play(FadeOut(error_text))

def generate_curriculum(prompt):
    parser = PydanticOutputParser(pydantic_object=Curriculum)
    
    prompt_template = PromptTemplate(
    input_variables=["topic"],
    template="""You are an expert curriculum maker for an AI video generator. Create a curriculum for a video lesson on {topic}. Provide a list of 5-7 topics, each with a title, animation type, and content. Ensure each topic includes all required properties.

    Animation types and content requirements:
    1. 'unit_circle': For topics related to the unit circle
       Content: Brief description of what to visualize (no formulas or equations)
       Example: {{"title": "Introduction to Unit Circle", "animation_type": "unit_circle", "content": "Demonstrate angles and points on the unit circle"}}

    2. 'graph': For topics that can be visualized with a graph
       Content: ONLY the function to graph, using 'x' as the variable. No explanations.
       Example: {{"title": "Linear Function", "animation_type": "graph", "content": "y = 2*x + 3"}}
       Example: {{"title": "Quadratic Function", "animation_type": "graph", "content": "y = x**2 - 4*x + 4"}}

    3. 'formula': For important equations or formulas
       Content: ONLY the formula, using LaTeX syntax. No explanations.
       Example: {{"title": "Pythagorean Theorem", "animation_type": "formula", "content": "a^2 + b^2 = c^2"}}

    4. 'text': For general explanations or concepts
       Content: A brief, clear explanation (1-2 sentences)
       Example: {{"title": "Properties of Logarithms", "animation_type": "text", "content": "Logarithms have three key properties: product rule, quotient rule, and power rule."}}

    Important:
    - For 'graph' type, include ONLY the function. Use 'x' as the variable and standard Python operators (**, *, /, +, -).
    - For 'formula' type, include ONLY the formula in LaTeX syntax.
    - Ensure the content is appropriate for the animation type and follows the examples closely.
    - Do not use placeholders like 'm' and 'b' in equations. Use specific numbers.

    Format your response as a valid JSON array of topics:
    [
      {{
        "title": "Topic Title",
        "animation_type": "animation_type",
        "content": "content"
      }},
      ...
    ]

    Generate a curriculum specifically for {topic}, adhering strictly to these guidelines and examples.
    """
    )
    
    prompt = prompt_template.format_prompt(topic=prompt)

    model = OpenAI(temperature=0.7)

    output = model(prompt.to_string())

    try:
        preprocessed_output = preprocess_json(output)
        topics_data = json.loads(preprocessed_output)
        
        curriculum = Curriculum(topics=[Topic(**topic) for topic in topics_data])
        
        print("Generated curriculum:")
        print(curriculum.model_dump_json(indent=2))
        return curriculum
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        print("Raw output from OpenAI:")
        print(output)
        print("Preprocessed output:")
        print(preprocessed_output)
    except Exception as e:
        print(f"Error creating curriculum: {e}")
        print("Raw output from OpenAI:")
        print(output)
    
    return Curriculum(topics=[Topic(title=f"Introduction to {prompt}", content=f"This lesson covers the basics of {prompt}", animation_type="text")])

def main():
    topic = input("Enter a topic for the educational video: ")
    curriculum = generate_curriculum(topic)
    
    print("Generated curriculum:")
    print(curriculum.model_dump_json(indent=2))  
    
    try:
        config.output_file = f"{topic.replace(' ', '_')}_lesson"
        config.quality = "medium_quality"
        config.preview = True
        
        scene = AIVideoEducationVisualizer(curriculum, topic)
        scene.render()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
