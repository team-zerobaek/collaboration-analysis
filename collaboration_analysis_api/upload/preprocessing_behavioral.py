import pandas as pd
import re
import base64
import os
import math
from collections import defaultdict
import networkx as nx
import numpy as np

def process_transcripts(contents, filenames):
    transcriptions = []
    for content, filename in zip(contents, filenames):
        content_type, content_string = content.split(',')
        decoded = base64.b64decode(content_string)
        transcriptions.append(decoded.decode('utf-8'))
    return transcriptions

def extract_speaker_turns(data):
    lines = data.split('\n')
    speakers = []
    texts = []
    word_counts = []

    current_speaker = None
    current_text = []

    for line in lines:
        speaker_match = re.match(r'Speaker (SPEAKER_\d+)', line)
        if speaker_match:
            if current_speaker:
                full_text = ' '.join(current_text).strip()
                speakers.append(current_speaker)
                texts.append(full_text)
                word_counts.append(len(full_text.split()))
                current_text = []
            current_speaker = speaker_match.group(1)
        else:
            current_text.append(line.strip())

    # Append the last speaker's text
    if current_speaker:
        full_text = ' '.join(current_text).strip()
        speakers.append(current_speaker)
        texts.append(full_text)
        word_counts.append(len(full_text.split()))

    df = pd.DataFrame({'Speaker': speakers, 'Text': texts, 'Word_Count': word_counts})
    return df

def create_dataset(dfs, project_number):
    dataset = []
    for i, df in enumerate(dfs):
        df['meeting_number'] = i + 1  # Add meeting number
        df['project'] = project_number  # Add project number
        speaker_word_counts = df.groupby('Speaker')['Word_Count'].sum().to_dict()
        total_words = df['Word_Count'].sum()
        for speaker, word_count in speaker_word_counts.items():
            speaker_number = int(speaker.split('_')[1])
            duration = extract_last_time_in_minutes(df['Text'].iloc[-1])
            if duration is None:
                duration = 0
            else:
                duration /= 60  # Convert to hours
            normalized_speech_frequency = word_count / duration if duration > 0 else 0
            dataset.append({
                'id': f'{project_number}_{i+1}_SPEAKER_{str(speaker_number).zfill(2)}',
                'project': project_number,
                'meeting_number': i + 1,
                'speaker_number': speaker_number,
                'speech_frequency': word_count,
                'total_words': total_words,
                'duration': duration,
                'normalized_speech_frequency': normalized_speech_frequency,
                'overall_collaboration_score': -1,  # Default value
                'individual_collaboration_score': -1  # Default value
            })
    created_dataset_df = pd.DataFrame(dataset)
    created_dataset_df.to_csv('/app/data/created_dataset.csv', index=False)
    return created_dataset_df

def extract_last_time_in_minutes(text):
    time_pattern_hms = re.compile(r'\b\d{1,2}:\d{2}:\d{2}\b')
    time_pattern_ms = re.compile(r'\b\d{2}:\d{2}\b')
    times_hms = time_pattern_hms.findall(text)
    times_ms = time_pattern_ms.findall(text)

    if not times_hms and not times_ms:
        return None
    if times_hms:
        last_time = times_hms[-1]
        hours, minutes, seconds = map(int, last_time.split(':'))
    else:
        last_time = times_ms[-1]
        hours = 0
        minutes, seconds = map(int, last_time.split(':'))
    total_minutes = hours * 60 + minutes
    if seconds > 0:
        total_minutes += math.ceil(seconds / 60)

    return total_minutes

def process_files_in_directory(transcriptions):
    durations = []
    for text in transcriptions:
        minutes = extract_last_time_in_minutes(text)
        durations.append(minutes)
    return durations

def compute_interaction_frequency(df, project_number):
    interaction_counts = defaultdict(lambda: defaultdict(int))
    interaction_records = []
    for i in range(len(df)):
        prev_speaker = df.iloc[i]['Speaker']
        if i < len(df) - 1:
            next_speaker = df.iloc[i+1]['Speaker']
        else:
            next_speaker = df.iloc[i]['Speaker']
        interaction_counts[prev_speaker][next_speaker] += 1
    for prev_speaker, next_speakers in interaction_counts.items():
        for next_speaker, count in next_speakers.items():
            interaction_records.append({
                'id': f'{project_number}_{df["meeting_number"].iloc[0]}_SPEAKER_{str(prev_speaker.split("_")[1]).zfill(2)}',
                'project': project_number,
                'meeting_number': df['meeting_number'].iloc[0],
                'speaker_id': int(prev_speaker.split('_')[1]),
                'next_speaker_id': int(next_speaker.split('_')[1]),
                'count': count
            })
    return pd.DataFrame(interaction_records)

def generate_all_pairs(interaction_records, dataset):
    all_pairs = []
    for (project, meeting), group in dataset.groupby(['project', 'meeting_number']):
        speakers = group['speaker_number'].unique()
        for speaker1 in speakers:
            for speaker2 in speakers:
                if not interaction_records[
                    (interaction_records['project'] == project) &
                    (interaction_records['meeting_number'] == meeting) &
                    (interaction_records['speaker_id'] == speaker1) &
                    (interaction_records['next_speaker_id'] == speaker2)
                ].empty:
                    continue
                all_pairs.append({
                    'id': f'{project}_{meeting}_SPEAKER_{str(speaker1).zfill(2)}',
                    'project': project,
                    'meeting_number': meeting,
                    'speaker_id': speaker1,
                    'next_speaker_id': speaker2,
                    'count': 0
                })
    return pd.DataFrame(all_pairs)

def compute_density(G):
    num_nodes = len(G)
    if num_nodes < 2:
        return 0
    possible_edges = num_nodes * (num_nodes - 1)  # For directed graph
    actual_edges = sum(1 for u, v, data in G.edges(data=True) if u != v and data['weight'] > 0)
    return actual_edges / possible_edges

def weighted_density(G):
    if len(G) == 0:
        return 0
    total_weight = sum(data['weight'] for u, v, data in G.edges(data=True))
    num_nodes = len(G)
    max_weight = max(data['weight'] for u, v, data in G.edges(data=True))
    possible_edges = num_nodes * (num_nodes)  # For directed graph
    return total_weight / (possible_edges * max_weight) if possible_edges > 0 else 0

def compute_centralities(df, combined_dataset):
    G = nx.DiGraph()
    for i in range(len(df)):
        prev_speaker = df.iloc[i]['Speaker']
        if i < len(df) - 1:
            next_speaker = df.iloc[i+1]['Speaker']
        else:
            next_speaker = df.iloc[i]['Speaker']  # Self-interaction if last speaker
        if prev_speaker != next_speaker and df.iloc[i]['Text'].strip() != '':
            combined_row = combined_dataset[
                (combined_dataset['project'] == df['project'].iloc[0]) &
                (combined_dataset['meeting_number'] == df['meeting_number'].iloc[0]) &
                (combined_dataset['speaker_id'] == int(prev_speaker.split('_')[1])) &
                (combined_dataset['next_speaker_id'] == int(next_speaker.split('_')[1]))
            ]
            if not combined_row.empty:
                weight = combined_row['count'].values[0]
                if G.has_edge(prev_speaker, next_speaker):
                    G[prev_speaker][next_speaker]['weight'] += weight
                else:
                    G.add_edge(prev_speaker, next_speaker, weight=weight)
    if len(G) == 0:
        centralities = {
            'degree_centrality': {},
            'indegree_centrality': {},
            'outdegree_centrality': {},
            'betweenness_centrality': {},
            'closeness_centrality': {},
            'eigenvector_centrality': {},
            'pagerank': {}
        }
    else:
        try:
            eigenvector_cent = nx.eigenvector_centrality(G, max_iter=2000, weight='weight')
        except nx.PowerIterationFailedConvergence:
            eigenvector_cent = {node: 0 for node in G.nodes()}
        num_nodes = len(G)
        possible_edges = num_nodes * (num_nodes - 1)
        max_weight = max(data['weight'] for u, v, data in G.edges(data=True) if u != v)
        centralities = {
            'degree_centrality': {k: v / (possible_edges * max_weight) for k, v in dict(G.degree(weight='weight')).items()},
            'indegree_centrality': {k: v / (possible_edges * max_weight) for k, v in dict(G.in_degree(weight='weight')).items()},
            'outdegree_centrality': {k: v / (possible_edges * max_weight) for k, v in dict(G.out_degree(weight='weight')).items()},
            'betweenness_centrality': {k: v / (possible_edges * max_weight) for k, v in nx.betweenness_centrality(G, weight='weight').items()},
            'closeness_centrality': {k: v / (possible_edges * max_weight) for k, v in nx.closeness_centrality(G, distance='weight').items()},
            'eigenvector_centrality': {k: v / (possible_edges * max_weight) for k, v in eigenvector_cent.items()},
            'pagerank': {k: v / (possible_edges * max_weight) for k, v in nx.pagerank(G, weight='weight').items()}
        }
    return centralities

def gini_coefficient(x):
    x = np.array(x, dtype=np.float64)
    if np.amin(x) < 0:
        x -= np.amin(x)  # values cannot be negative
    x += 0.0000001  # values cannot be 0
    x = np.sort(x)  # values must be sorted
    index = np.arange(1, x.shape[0] + 1)  # index per array element
    n = x.shape[0]
    return ((np.sum((2 * index - n - 1) * x)) / (n * np.sum(x)))

def compute_gini(df):
    gini_values = []
    meetings = df['meeting_number'].unique()
    for meeting_number in meetings:
        meeting_data = df[df['meeting_number'] == meeting_number]
        interaction_counts = [
            meeting_data[meeting_data['speaker_id'] == speaker]['count'].sum()
            for speaker in meeting_data['speaker_id'].unique()
        ]
        gini_values.append(gini_coefficient(interaction_counts))
    return gini_values

def interaction_equality_index(x):
    x = np.array(x, dtype=np.float64)
    mean_x = np.mean(x)
    if mean_x == 0:
        return 0
    return 1 - (np.std(x) / mean_x)

def compute_equality_index(df):
    equality_index_values = []
    meetings = df['meeting_number'].unique()
    for meeting_number in meetings:
        meeting_data = df[df['meeting_number'] == meeting_number]
        interaction_counts = [
            meeting_data[meeting_data['speaker_id'] == speaker]['count'].sum()
            for speaker in meeting_data['speaker_id'].unique()
        ]
        equality_index_values.append(interaction_equality_index(interaction_counts))
    return equality_index_values

def process_and_save(dataset, filename):
    # Process the dataset and save it to the specified file
    dataset.to_csv(filename, index=False)

# Additional functions for saving intermediate data
def save_dataframe_to_csv(df, filename):
    df.to_csv(filename, index=False)
