import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import anthropic
import json
import re
from typing import Dict, List, Any, Optional
import spacy
# Keep the existing THEME settings
THEME = {
    'colors': ['#4C72B0', '#55A868', '#C44E52', '#8172B3', '#CCB974'],
    'background': '#ffffff',
    'text': '#2c3e50',
    'grid': '#ecf0f1',
    'font_sizes': {
        'title': 16,
        'label': 14,
        'tick': 12,
        'annotation': 11
    }
}

plt.rcParams.update({
    'axes.titlesize': THEME['font_sizes']['title'],
    'axes.labelsize': THEME['font_sizes']['label'],
    'xtick.labelsize': THEME['font_sizes']['tick'],
    'ytick.labelsize': THEME['font_sizes']['tick'],
    'grid.color': THEME['grid'],
    'grid.alpha': 0.5,
    'font.family': 'sans-serif',
    'figure.dpi': 100,
    'savefig.dpi': 300
})


class TextAnalyzer:
    def __init__(self):
        # Initialize spaCy
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            st.error("Please install spacy and download the English model")
            self.nlp = None
            
            
    def analyze_with_claude(self, text: str, api_key: str) -> Dict[str, Any]:
        client = anthropic.Anthropic(api_key=api_key)
        
        prompt = f"""
Analyze this text and extract the following information in JSON format:
{{
    "numbers": [
        {{
            "value": float,
            "type": "percentage|currency|large_number|number",
            "context": "What the number refers to in one sentence",
            "date": "YYYY-MM-DD",  // Optional, if number is associated with a specific date
            "sentence": "The full sentence where the number was mentioned"
        }}
    ],
    "dates": [
        {{
            "date": "YYYY-MM-DD",
            "context": "What the date refers to in one sentence",
            "sentence": "The full sentence where the date was mentioned"
        }}
    ],
    "categories": {{
        "category_name": count
    }}
}}
Guidelines:
1. Extract and normalize all dates to the format YYYY-MM-DD.
2. Identify numeric values and classify them as percentages, currencies, or general large numbers. Provide their context and sentence.
3. Extract categories and provide their counts.
4. Ignore generic phrases or unrelated data.
Text: {text}
"""

        try:
            response = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            return json.loads(response.content[0].text)
        except Exception as e:
            st.error(f"Claude analysis error: {str(e)}")
            return {}

    
            
    def _extract_numbers_regex(self, text: str) -> List[Dict[str, Any]]:
        """Enhanced number extraction using regex."""
        numbers = []
        
        # Define regex patterns for different number formats
        patterns = [
            # Currency with billions/millions
            (r'\$\s*(\d+(?:\.\d+)?)\s*[Bb]illion', 'currency', 1e9),
            (r'\$\s*(\d+(?:\.\d+)?)\s*[Bb]', 'currency', 1e9),
            (r'\$\s*(\d+(?:\.\d+)?)\s*[Mm]illion', 'currency', 1e6),
            (r'\$\s*(\d+(?:\.\d+)?)\s*[Mm]', 'currency', 1e6),
            
            # Regular currency
            (r'\$\s*(\d+(?:,\d{3})*(?:\.\d+)?)', 'currency', 1),
            
            # Percentages
            (r'(\d+(?:\.\d+)?)\s*%', 'percentage', 1),
            
            # Numbers with units
            (r'(\d+(?:\.\d+)?)\s*billion', 'large_number', 1e9),
            (r'(\d+(?:\.\d+)?)\s*million', 'large_number', 1e6),
            (r'(\d+(?:\.\d+)?)\s*[BbMm]', 'large_number', 1e9),
        ]
        
        # Extract sentences and process each one
        sentences = text.split('.')
        for sentence in sentences:
            # Find date context if any
            date_match = re.search(r'\(([^)]+)\)', sentence)
            date = None
            if date_match:
                try:
                    date = pd.to_datetime(date_match.group(1)).strftime('%Y-%m-%d')
                except:
                    pass
            
            for pattern, num_type, multiplier in patterns:
                matches = re.finditer(pattern, sentence)
                for match in matches:
                    try:
                        value_str = match.group(1).replace(',', '')
                        value = float(value_str) * multiplier
                        numbers.append({
                            "value": value,
                            "type": num_type,
                            "context": sentence.strip(),
                            "sentence": sentence.strip(),
                            "date": date
                        })
                    except:
                        continue
        
        return numbers

    def _extract_dates_regex(self, text: str) -> List[Dict[str, str]]:
        """Enhanced date extraction using regex."""
        dates = []
        
        # Define regex patterns for different date formats
        date_patterns = [
            (r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}',),
            (r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}',),
            (r'\d{4}-\d{2}-\d{2}',),
            (r'Q[1-4]\s+\d{4}',),
            (r'\(([^)]+)\)',),  # Dates in parentheses
        ]
        
        # Extract sentences and process each one
        sentences = text.split('.')
        for sentence in sentences:
            for pattern in date_patterns:
                matches = re.finditer(pattern[0], sentence)
                for match in matches:
                    try:
                        date_str = match.group()
                        # Handle quarter dates
                        if date_str.startswith('Q'):
                            quarter = int(date_str[1])
                            year = int(date_str[-4:])
                            month = (quarter - 1) * 3 + 1
                            date_str = f"{year}-{month:02d}-01"
                            
                        parsed_date = pd.to_datetime(date_str)
                        dates.append({
                            "date": parsed_date.strftime("%Y-%m-%d"),
                            "context": sentence.strip(),
                            "sentence": sentence.strip()
                        })
                    except:
                        continue
        
        return dates

    def _extract_categories_regex(self, text: str) -> Dict[str, int]:
        """Enhanced category extraction using regex."""
        categories = {}
        
        # Pattern for categories with percentages
        category_pattern = r'-\s*([^:]+):\s*(\d+(?:\.\d+)?)\s*%'
        
        matches = re.finditer(category_pattern, text)
        for match in matches:
            category = match.group(1).strip()
            value = float(match.group(2))
            categories[category] = value
            
        return categories

    def analyze_with_spacy_enhanced(self, text: str) -> Dict[str, Any]:
        """Enhanced spaCy analysis combined with regex."""
        if not self.nlp:
            return {}
            
        # Get basic spaCy analysis
        doc = self.nlp(text)
        
        # Get enhanced extractions using regex
        numbers = self._extract_numbers_regex(text)
        dates = self._extract_dates_regex(text)
        categories = self._extract_categories_regex(text)
        
        # Add any additional entities from spaCy
        for ent in doc.ents:
            if ent.label_ == "DATE":
                try:
                    parsed_date = pd.to_datetime(ent.text)
                    if not any(d['date'] == parsed_date.strftime("%Y-%m-%d") for d in dates):
                        dates.append({
                            "date": parsed_date.strftime("%Y-%m-%d"),
                            "context": ent.sent.text.strip(),
                            "sentence": ent.sent.text.strip()
                        })
                except:
                    continue
                    
        return {
            "numbers": numbers,
            "dates": dates,
            "categories": categories
        }

    

def generate_pie_chart(categories: Dict[str, int]) -> Optional[plt.Figure]:
    if not categories:
        return None
        
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(THEME['background'])
    ax.set_facecolor(THEME['background'])
    
    values = list(categories.values())
    labels = [f"{k}\n({v})" for k, v in categories.items()]
    
    patches, texts, autotexts = ax.pie(
        values,
        labels=labels,
        colors=THEME['colors'],
        autopct='%1.1f%%',
        pctdistance=0.85
    )
    
    plt.title('Category Distribution', pad=20, size=THEME['font_sizes']['title'])
    plt.tight_layout()
    return fig

def generate_line_chart(numbers: List[Dict[str, Any]]) -> Optional[plt.Figure]:
    time_series = [n for n in numbers if n.get('date')]
    if len(time_series) < 2:
        return None
        
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(THEME['background'])
    ax.set_facecolor(THEME['background'])
    
    df = pd.DataFrame(time_series)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    ax.plot(df['date'], df['value'],
            marker='o',
            color=THEME['colors'][0],
            linewidth=2,
            markersize=8)
    
    for x, y in zip(df['date'], df['value']):
        ax.annotate(f'{y:,.0f}',
                   (x, y),
                   textcoords="offset points",
                   xytext=(0, 10),
                   ha='center',
                   fontsize=THEME['font_sizes']['annotation'])
    
    plt.xticks(rotation=45)
    plt.title('Values Over Time', pad=20, size=THEME['font_sizes']['title'])
    plt.tight_layout()
    return fig

def generate_timeline(dates: List[Dict[str, str]]) -> Optional[plt.Figure]:
    if len(dates) < 2:
        return None
        
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(THEME['background'])
    ax.set_facecolor(THEME['background'])
    
    timeline_data = []
    for date_entry in dates:
        try:
            date = pd.to_datetime(date_entry['date'])
            timeline_data.append({
                'date': date,
                'event': date_entry['context']
            })
        except:
            continue
            
    if not timeline_data:
        return None
        
    timeline_data.sort(key=lambda x: x['date'])
    
    ax.scatter(range(len(timeline_data)), [0]*len(timeline_data),
              color=THEME['colors'][0], s=100, zorder=2)
    
    ax.plot(range(len(timeline_data)), [0]*len(timeline_data),
            color=THEME['colors'][0], alpha=0.3, zorder=1)
    
    for i, event in enumerate(timeline_data):
        ax.annotate(
            f"{event['date'].strftime('%Y-%m-%d')}\n{event['event']}",
            xy=(i, 0),
            xytext=(0, 10 + (i % 2) * 20),
            textcoords='offset points',
            ha='center',
            va='bottom',
            fontsize=THEME['font_sizes']['annotation']
        )
    
    ax.set_yticks([])
    plt.title('Timeline of Events', pad=20, size=THEME['font_sizes']['title'])
    plt.tight_layout()
    return fig

def format_value(value: float) -> str:
    """Format large numbers into readable strings."""
    if value >= 1e9:
        return f'${value/1e9:.0f}B'
    elif value >= 1e6:
        return f'${value/1e6:.0f}M'
    elif value >= 1e3:
        return f'${value/1e3:.0f}K'
    else:
        return f'${value:.0f}'

def clean_context(context: str) -> str:
    """Clean and shorten context strings."""
    # Remove common prefixes
    context = re.sub(r'^Q[1-4] revenue:', '', context)
    context = re.sub(r'^revenue:', '', context)
    
    # Remove date in parentheses
    context = re.sub(r'\([^)]+\)', '', context)
    
    # Trim and clean
    context = context.strip()
    if len(context) > 20:
        context = context[:18] + '...'
    
    return context

def generate_bar_chart(numbers: List[Dict[str, Any]]) -> Optional[plt.Figure]:
    if not numbers:
        return None
        
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(THEME['background'])
    ax.set_facecolor(THEME['background'])
    
    # Clean and prepare data
    values = []
    contexts = []
    seen_contexts = set()
    
    for n in numbers:
        context = clean_context(n['context'])
        if context not in seen_contexts:  # Avoid duplicates
            values.append(n['value'])
            contexts.append(context)
            seen_contexts.add(context)
    
    # Create bars
    bars = ax.bar(range(len(values)), values, color=THEME['colors'][0])
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                format_value(height),
                ha='center', va='bottom',
                fontsize=THEME['font_sizes']['annotation'])
    
    # Format y-axis labels
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_value(x)))
    
    # Customize x-axis
    plt.xticks(range(len(contexts)), contexts, rotation=45, ha='right')
    
    # Add title and adjust layout
    plt.title('Numeric Values', pad=20, size=THEME['font_sizes']['title'])
    plt.grid(True, axis='y', alpha=0.3)
    
    # Ensure nothing is cut off
    plt.tight_layout()
    
    return fig

def main():
    st.title("Infographics Generator")
    
    # Input section
    st.header("Input")
    text = st.text_area("Enter your text:", height=200)
    num_graphics = st.number_input("Number of infographics required:", min_value=1, max_value=4, value=2)
    
    # Analysis method selection
    analysis_method = st.radio(
        "Select text analysis method:",
        ["Claude", "Enhanced spaCy","Compare All"]
    )
    
    # Claude API key input if needed
    api_key = None
    if analysis_method in ["Claude", "Compare All"]:
        api_key = st.text_input("Enter Claude API key:", type="password")
    
    if st.button("Generate Infographics"):
        if not text:
            st.error("Please enter some text.")
            return
            
        if analysis_method in ["Claude", "Compare All"] and not api_key:
            st.error("Please enter your Claude API key.")
            return
            
        analyzer = TextAnalyzer()
        
        # Compare all methods if selected
        if analysis_method == "Compare All":
            st.header("Comparison of Analysis Methods")
            
            # Get results from each method
            claude_data = analyzer.analyze_with_claude(text, api_key)
            spacy_data = analyzer.analyze_with_spacy_enhanced(text)
            
            methods = {
                "Claude": claude_data,
                "Enhanced spaCy": spacy_data
            }
            
            # Show results for each method
            for method_name, data in methods.items():
                st.subheader(f"\nResults from {method_name}")
                
                if not data:
                    st.warning(f"No data extracted using {method_name}")
                    continue
                
                # Create visualizations for this method
                visualizations = []
                
                if data.get('categories'):
                    fig = generate_pie_chart(data['categories'])
                    if fig:
                        visualizations.append((f"Category Distribution ({method_name})", fig))
                        
                if data.get('numbers'):
                    fig = generate_bar_chart(data['numbers'])
                    if fig:
                        visualizations.append((f"Numeric Values ({method_name})", fig))
                    
                time_series = [n for n in data.get('numbers', []) if n.get('date')]
                if len(time_series) >= 2:
                    fig = generate_line_chart(time_series)
                    if fig:
                        visualizations.append((f"Time Series ({method_name})", fig))
                        
                if data.get('dates'):
                    fig = generate_timeline(data['dates'])
                    if fig:
                        visualizations.append((f"Timeline ({method_name})", fig))
                
                # Display visualizations for this method
                if not visualizations:
                    st.warning(f"No visualizable data found using {method_name}")
                else:
                    for title, fig in visualizations[:num_graphics]:
                        st.markdown(f"**{title}**")
                        st.pyplot(fig)
                        plt.close(fig)
                        
                st.markdown("---")  # Add separator between methods
                
        else:
            # Single method analysis
            data = None
            if analysis_method == "Claude":
                data = analyzer.analyze_with_claude(text, api_key)
            elif analysis_method == "Enhanced spaCy":
                data = analyzer.analyze_with_spacy_enhanced(text)
            else:  # GPT-2
                data = analyzer.analyze_with_gpt2(text)
                
            if not data:
                st.error("No data could be extracted from the text.")
                return
                
            # Create visualizations
            visualizations = []
            
            if data.get('categories'):
                fig = generate_pie_chart(data['categories'])
                if fig:
                    visualizations.append(("Category Distribution", fig))
                    
            if data.get('numbers'):
                fig = generate_bar_chart(data['numbers'])
                if fig:
                    visualizations.append(("Numeric Values", fig))
                
            time_series = [n for n in data.get('numbers', []) if n.get('date')]
            if len(time_series) >= 2:
                fig = generate_line_chart(time_series)
                if fig:
                    visualizations.append(("Time Series", fig))
                    
            if data.get('dates'):
                fig = generate_timeline(data['dates'])
                if fig:
                    visualizations.append(("Timeline", fig))
            
            # Display visualizations
            if not visualizations:
                st.warning("No visualizable data found in the text.")
            else:
                for title, fig in visualizations[:num_graphics]:
                    st.subheader(title)
                    st.pyplot(fig)
                    plt.close(fig)

if __name__ == "__main__":
    main()