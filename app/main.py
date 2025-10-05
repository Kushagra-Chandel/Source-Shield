import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

# Import our modules
from config import config
from database import db_manager
from scraping.extractor import content_extractor
from nlp.classifier import fact_opinion_classifier
from nlp.sentiment_bias import sentiment_bias_analyzer
from nlp.similarity import similarity_analyzer
from llm.source_comparison import source_comparison_analyzer
from utils.helpers import is_valid_url, format_timestamp, setup_logging

# Setup logging
setup_logging()

def main():
    st.set_page_config(
        page_title="SourceShield - News Analysis Tool",
        page_icon="🛡️",
        layout="wide"
    )
    
    st.title("🛡️ SourceShield - News Bias & Fact Analysis")
    st.markdown("Analyze news articles for bias, sentiment, and fact vs opinion classification")
    
    # Sidebar
    st.sidebar.title("Analysis Options")
    analysis_type = st.sidebar.selectbox(
        "Choose Analysis Type",
        ["Single Article Analysis", "Multi-Source Comparison", "Historical Analysis"]
    )
    
    if analysis_type == "Single Article Analysis":
        single_article_analysis()
    elif analysis_type == "Multi-Source Comparison":
        multi_source_comparison()
    else:
        historical_analysis()

def single_article_analysis():
    st.header("Single Article Analysis")
    
    # Input options
    input_type = st.radio("Input Type", ["URL", "Direct Text"])
    
    if input_type == "URL":
        url = st.text_input("Enter News Article URL:", placeholder="https://example.com/news-article")
        
        if st.button("Analyze Article", type="primary"):
            if url and is_valid_url(url):
                analyze_single_source(url, None)
            else:
                st.error("Please enter a valid URL")
    
    else:
        text_input = st.text_area("Enter Text to Analyze:", height=200, placeholder="Paste your news text here...")
        
        if st.button("Analyze Text", type="primary"):
            if text_input.strip():
                analyze_single_source(None, text_input)
            else:
                st.error("Please enter some text to analyze")

def analyze_single_source(url, text):
    """Analyze a single source (URL or text)"""
    with st.spinner("Extracting and analyzing content..."):
        
        # Extract content
        if url:
            content_data = content_extractor.extract_from_url(url)
        else:
            content_data = content_extractor.extract_from_text(text)
        
        if "error" in content_data:
            st.error(f"Content Extraction Error: {content_data['error']}")
            
            # Show suggestion if available
            if "suggestion" in content_data:
                st.info(f"💡 Suggestion: {content_data['suggestion']}")
            return
        
        # Display content info
        st.subheader("📄 Article Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Word Count", content_data.get('word_count', 0))
        with col2:
            st.metric("Domain", content_data.get('domain', 'N/A'))
        with col3:
            if content_data.get('publish_date'):
                st.metric("Published", content_data['publish_date'].strftime("%Y-%m-%d"))
            else:
                st.metric("Published", "Unknown")
        
        if content_data.get('title'):
            st.write(f"**Title:** {content_data['title']}")
        
        # Perform analysis
        article_text = content_data.get('content', '')
        
        if not article_text or len(article_text.strip()) < 20:
            st.error("Insufficient content found for meaningful analysis")
            return
        
        # Run all analyses with error handling
        with st.spinner("Running comprehensive analysis..."):
            
            # 1. Fact vs Opinion Classification
            try:
                fact_opinion_result = fact_opinion_classifier.classify_text(article_text)
            except Exception as e:
                st.warning(f"Fact/Opinion analysis failed: {str(e)}")
                fact_opinion_result = {"error": str(e)}
            
            # 2. Sentiment and Bias Analysis
            try:
                bias_result = sentiment_bias_analyzer.comprehensive_bias_analysis(article_text)
            except Exception as e:
                st.warning(f"Bias analysis failed: {str(e)}")
                bias_result = {"error": str(e)}
            
            # 3. LLM Analysis (if available)
            try:
                llm_result = source_comparison_analyzer.comprehensive_llm_analysis(article_text)
            except Exception as e:
                st.warning(f"AI analysis failed: {str(e)}")
                llm_result = {"error": str(e)}
        
        # Display results
        display_analysis_results(fact_opinion_result, bias_result, llm_result)
        
        # Save to database (only if at least one analysis succeeded)
        if not all("error" in result for result in [fact_opinion_result, bias_result, llm_result]):
            analysis_data = {
                "fact_opinion": fact_opinion_result,
                "bias_analysis": bias_result,
                "llm_analysis": llm_result,
                "timestamp": datetime.now()
            }
            
            try:
                db_manager.save_analysis(
                    url or "direct_input",
                    article_text,
                    analysis_data
                )
            except Exception as e:
                st.warning(f"Failed to save analysis to database: {str(e)}")

def display_analysis_results(fact_opinion_result, bias_result, llm_result):
    """Display comprehensive analysis results"""
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🎯 Fact vs Opinion", "⚖️ Bias Analysis", "🤖 AI Analysis"])
    
    with tab1:
        st.subheader("Analysis Overview")
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'statistics' in fact_opinion_result:
                fact_pct = fact_opinion_result['statistics'].get('fact_percentage', 0)
                st.metric("Fact Percentage", f"{fact_pct}%")
        
        with col2:
            if 'sentiment_analysis' in bias_result and 'confidence' in bias_result['sentiment_analysis']:
                sentiment_conf = bias_result['sentiment_analysis']['confidence']
                st.metric("Sentiment Confidence", f"{sentiment_conf:.2f}")
        
        with col3:
            if 'political_bias' in bias_result and 'confidence' in bias_result['political_bias']:
                bias_conf = bias_result['political_bias']['confidence']
                st.metric("Bias Confidence", f"{bias_conf:.2f}")
        
        with col4:
            overall_score = calculate_overall_credibility(fact_opinion_result, bias_result)
            st.metric("Credibility Score", f"{overall_score:.2f}")
        
        # Visual summary
        create_overview_charts(fact_opinion_result, bias_result)
    
    with tab2:
        st.subheader("Fact vs Opinion Analysis")
        display_fact_opinion_results(fact_opinion_result)
    
    with tab3:
        st.subheader("Bias Analysis")
        display_bias_results(bias_result)
    
    with tab4:
        st.subheader("AI-Powered Analysis")
        display_llm_results(llm_result)

def create_overview_charts(fact_opinion_result, bias_result):
    """Create overview visualizations"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Fact vs Opinion pie chart
        if 'statistics' in fact_opinion_result:
            stats = fact_opinion_result['statistics']
            
            fig = px.pie(
                values=[stats.get('fact_percentage', 0), stats.get('opinion_percentage', 0)],
                names=['Facts', 'Opinions'],
                title="Fact vs Opinion Distribution",
                color_discrete_sequence=['#1f77b4', '#ff7f0e']
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Bias indicators
        if 'political_bias' in bias_result:
            political_bias = bias_result['political_bias'].get('political_bias', 'neutral')
            confidence = bias_result['political_bias'].get('confidence', 0)
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = confidence,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': f"Political Bias: {political_bias.replace('_', ' ').title()}"},
                gauge = {
                    'axis': {'range': [None, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 0.5], 'color': "lightgray"},
                        {'range': [0.5, 0.8], 'color': "yellow"},
                        {'range': [0.8, 1], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.9
                    }
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

def display_fact_opinion_results(fact_opinion_result):
    """Display fact vs opinion analysis results"""
    
    if 'error' in fact_opinion_result:
        st.error(fact_opinion_result['error'])
        return
    
    # Overall classification
    overall_class = fact_opinion_result.get('overall_classification', 'unknown')
    st.info(f"**Overall Classification:** {overall_class.replace('_', ' ').title()}")
    
    # Statistics
    if 'statistics' in fact_opinion_result:
        stats = fact_opinion_result['statistics']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Sentences", stats.get('total_sentences', 0))
        with col2:
            st.metric("Fact Sentences", stats.get('fact_sentences', 0))
        with col3:
            st.metric("Opinion Sentences", stats.get('opinion_sentences', 0))
    
    # Sentence breakdown
    if 'sentence_breakdown' in fact_opinion_result:
        st.subheader("Sentence-by-Sentence Analysis")
        
        breakdown = fact_opinion_result['sentence_breakdown']
        
        # Create DataFrame for display
        df_data = []
        for i, item in enumerate(breakdown[:10]):  # Show first 10 sentences
            df_data.append({
                'Sentence': item['sentence'][:100] + "..." if len(item['sentence']) > 100 else item['sentence'],
                'Classification': item['classification'].title(),
                'Confidence': f"{item['confidence']:.2f}"
            })
        
        if df_data:
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True)

def display_bias_results(bias_result):
    """Display bias analysis results"""
    
    if 'error' in bias_result:
        st.error(bias_result['error'])
        return
    
    # Sentiment Analysis
    if 'sentiment_analysis' in bias_result:
        st.subheader("Sentiment Analysis")
        sentiment = bias_result['sentiment_analysis']
        
        if 'primary_sentiment' in sentiment:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Primary Sentiment", sentiment['primary_sentiment'].title())
            with col2:
                st.metric("Confidence", f"{sentiment.get('confidence', 0):.2f}")
            
            # Sentiment scores chart
            if 'all_scores' in sentiment:
                scores_df = pd.DataFrame(list(sentiment['all_scores'].items()), 
                                       columns=['Sentiment', 'Score'])
                fig = px.bar(scores_df, x='Sentiment', y='Score', 
                           title="Sentiment Scores")
                st.plotly_chart(fig, use_container_width=True)
    
    # Political Bias
    if 'political_bias' in bias_result:
        st.subheader("Political Bias Analysis")
        political = bias_result['political_bias']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Political Lean", political.get('political_bias', 'unknown').replace('_', ' ').title())
        with col2:
            st.metric("Confidence", f"{political.get('confidence', 0):.2f}")
        
        # Keyword counts
        if 'keyword_counts' in political:
            counts = political['keyword_counts']
            counts_df = pd.DataFrame(list(counts.items()), 
                                   columns=['Bias Type', 'Count'])
            fig = px.bar(counts_df, x='Bias Type', y='Count', 
                       title="Political Bias Keyword Counts")
            st.plotly_chart(fig, use_container_width=True)
    
    # Emotional Bias
    if 'emotional_bias' in bias_result:
        st.subheader("Emotional Bias Analysis")
        emotional = bias_result['emotional_bias']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Emotional Level", emotional.get('emotional_bias', 'unknown').replace('_', ' ').title())
        with col2:
            st.metric("Emotional Score", f"{emotional.get('emotional_score', 0):.2f}")

def display_llm_results(llm_result):
    """Display LLM analysis results"""
    
    if not llm_result or 'error' in str(llm_result):
        st.warning("LLM analysis not available. Please check your OpenAI API key.")
        return
    
    # Bias Analysis
    if 'bias_analysis' in llm_result:
        st.subheader("AI Bias Analysis")
        bias_analysis = llm_result['bias_analysis']
        
        if 'error' not in bias_analysis:
            if isinstance(bias_analysis, dict):
                for key, value in bias_analysis.items():
                    if key != 'raw_response':
                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")
            else:
                st.write(bias_analysis)
    
    # Fact vs Opinion Analysis
    if 'fact_opinion_analysis' in llm_result:
        st.subheader("AI Fact vs Opinion Analysis")
        fact_opinion = llm_result['fact_opinion_analysis']
        
        if 'error' not in fact_opinion:
            if isinstance(fact_opinion, dict):
                for key, value in fact_opinion.items():
                    if key not in ['raw_response', 'analysis']:
                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")
            else:
                st.write(fact_opinion)

def multi_source_comparison():
    """Multi-source comparison interface"""
    st.header("Multi-Source Comparison")
    
    st.info("Compare multiple sources covering the same story to identify bias and perspective differences.")
    
    # Input for multiple sources
    num_sources = st.slider("Number of sources to compare", 2, 5, 2)
    
    sources = []
    for i in range(num_sources):
        st.subheader(f"Source {i+1}")
        input_type = st.radio(f"Input type for source {i+1}", ["URL", "Text"], key=f"input_type_{i}")
        
        if input_type == "URL":
            url = st.text_input(f"URL for source {i+1}", key=f"url_{i}")
            if url:
                sources.append(("url", url))
        else:
            text = st.text_area(f"Text for source {i+1}", key=f"text_{i}")
            if text:
                sources.append(("text", text))
    
    if st.button("Compare Sources", type="primary"):
        if len(sources) >= 2:
            compare_multiple_sources(sources)
        else:
            st.error("Please provide at least 2 sources to compare")

def compare_multiple_sources(sources):
    """Compare multiple sources"""
    with st.spinner("Extracting and analyzing all sources..."):
        
        source_contents = []
        source_analyses = []
        
        # Extract content from all sources
        for i, (source_type, source_data) in enumerate(sources):
            if source_type == "url":
                content_data = content_extractor.extract_from_url(source_data)
            else:
                content_data = content_extractor.extract_from_text(source_data)
            
            if "error" not in content_data:
                source_contents.append(content_data)
                
                # Analyze each source
                article_text = content_data.get('content', '')
                analysis = {
                    'fact_opinion': fact_opinion_classifier.classify_text(article_text),
                    'bias': sentiment_bias_analyzer.comprehensive_bias_analysis(article_text)
                }
                source_analyses.append(analysis)
        
        if len(source_contents) < 2:
            st.error("Could not extract content from enough sources")
            return
        
        # Display comparison results
        display_source_comparison(source_contents, source_analyses)

def display_source_comparison(source_contents, source_analyses):
    """Display comparison results"""
    
    st.subheader("Source Comparison Results")
    
    # Overview comparison table
    comparison_data = []
    for i, (content, analysis) in enumerate(zip(source_contents, source_analyses)):
        
        fact_pct = 0
        if 'statistics' in analysis['fact_opinion']:
            fact_pct = analysis['fact_opinion']['statistics'].get('fact_percentage', 0)
        
        political_bias = "unknown"
        if 'political_bias' in analysis['bias']:
            political_bias = analysis['bias']['political_bias'].get('political_bias', 'unknown')
        
        sentiment = "unknown"
        if 'sentiment_analysis' in analysis['bias']:
            sentiment = analysis['bias']['sentiment_analysis'].get('primary_sentiment', 'unknown')
        
        comparison_data.append({
            'Source': f"Source {i+1}",
            'Domain': content.get('domain', 'N/A'),
            'Word Count': content.get('word_count', 0),
            'Fact %': f"{fact_pct}%",
            'Political Bias': political_bias.replace('_', ' ').title(),
            'Sentiment': sentiment.title()
        })
    
    df = pd.DataFrame(comparison_data)
    st.dataframe(df, use_container_width=True)
    
    # Similarity analysis
    if len(source_contents) >= 2:
        st.subheader("Content Similarity Analysis")
        
        texts = [content.get('content', '') for content in source_contents]
        similarity_result = similarity_analyzer.detect_content_overlap(texts)
        
        if 'error' not in similarity_result:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Average Similarity", f"{similarity_result.get('average_similarity', 0):.2f}")
            
            with col2:
                highest = similarity_result.get('highest_similarity', {})
                if highest:
                    pair = highest.get('article_pair', [0, 1])
                    similarity = highest.get('similarity', 0)
                    st.metric("Highest Similarity", f"Sources {pair[0]+1} & {pair[1]+1}: {similarity:.2f}")

def historical_analysis():
    """Historical analysis interface"""
    st.header("Historical Analysis")
    
    # Get recent analyses from database
    recent_analyses = db_manager.get_recent_analyses(20)
    
    if not recent_analyses:
        st.info("No previous analyses found. Analyze some articles first!")
        return
    
    # Display recent analyses
    st.subheader("Recent Analyses")
    
    analyses_data = []
    for analysis in recent_analyses:
        analyses_data.append({
            'Timestamp': format_timestamp(analysis.get('timestamp')),
            'URL/Source': analysis.get('url', 'N/A')[:50] + "..." if len(analysis.get('url', '')) > 50 else analysis.get('url', 'N/A'),
            'Content Preview': analysis.get('content', '')[:100] + "..." if len(analysis.get('content', '')) > 100 else analysis.get('content', '')
        })
    
    if analyses_data:
        df = pd.DataFrame(analyses_data)
        st.dataframe(df, use_container_width=True)
    
    # Analytics
    st.subheader("Analysis Trends")
    
    # Create trend charts based on historical data
    if len(recent_analyses) > 1:
        # Extract trend data
        dates = [analysis.get('timestamp', datetime.now()) for analysis in recent_analyses]
        # Add more trend analysis here based on your needs
        
        st.info("Trend analysis will be implemented based on accumulated data")

def calculate_overall_credibility(fact_opinion_result, bias_result):
    """Calculate overall credibility score"""
    scores = []
    
    # Fact percentage contribution
    if 'statistics' in fact_opinion_result:
        fact_pct = fact_opinion_result['statistics'].get('fact_percentage', 0)
        scores.append(fact_pct / 100)
    
    # Sentiment neutrality contribution
    if 'sentiment_analysis' in bias_result:
        sentiment = bias_result['sentiment_analysis'].get('primary_sentiment', '')
        if sentiment == 'neutral':
            scores.append(0.8)
        else:
            scores.append(0.5)
    
    # Political bias neutrality contribution
    if 'political_bias' in bias_result:
        political = bias_result['political_bias'].get('political_bias', '')
        if political in ['neutral', 'balanced']:
            scores.append(0.8)
        else:
            scores.append(0.4)
    
    return sum(scores) / len(scores) if scores else 0.5

if __name__ == "__main__":
    main()