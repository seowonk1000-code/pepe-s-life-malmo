import torch
import torch.nn as nn
import random
import math

class HumanScaleConnectome(nn.Module):
    """
    An advanced biological-style connectome model simulating a human-like mind named Pepe.
    Features:
    - Directed synaptic adjacency connectome matrix
    - Dynamic Hebbian learning (STDP/Plasticity) for permanent memory retention
    - Internal chemical tracking (Dopamine, Serotonin, Noradrenaline) mapping human emotions
    - Synaptic exhaustion/energy limits to prevent seizure-like infinite loops
    """
    def __init__(self, num_neurons=2000, inputs_dim=35, outputs_dim=35):
        super(HumanScaleConnectome, self).__init__()
        self.num_neurons = num_neurons
        self.inputs_dim = inputs_dim
        self.outputs_dim = outputs_dim
        
        # 1. Structural Connectome Assets
        self.input_weights = nn.Parameter(torch.randn(num_neurons, inputs_dim) * 0.1)
        self.synaptic_matrix = nn.Parameter(torch.randn(num_neurons, num_neurons) * (1.0 / num_neurons))
        self.output_weights = nn.Parameter(torch.randn(outputs_dim, num_neurons) * 0.1)
        
        # 2. Neurotransmitter levels per neuron
        self.neuron_dopamine = nn.Parameter(torch.rand(num_neurons))
        self.neuron_serotonin = nn.Parameter(torch.rand(num_neurons))
        self.neuron_noradrenaline = nn.Parameter(torch.rand(num_neurons))
        
        # 3. Human Biological Limitations (Energy and Thresholds)
        self.base_threshold = 0.4
        self.register_buffer('neuron_energy', torch.ones(num_neurons)) # 1.0 = Fully rested cell
        
        # 4. Specialized Human Language and Emotional Sectors
        self.wernicke_start, self.wernicke_end = 0, int(num_neurons * 0.3)      # Language input processing
        self.broca_start, self.broca_end = int(num_neurons * 0.3), int(num_neurons * 0.5) # Speech creation
        self.limbic_start, self.limbic_end = int(num_neurons * 0.5), int(num_neurons * 0.7) # Emotional/Crush processing
        
    def forward(self, sensory_inputs, current_state=None):
        if current_state is None:
            current_state = torch.zeros(self.num_neurons)
            
        # Biological Energy Recovery (resting cells slowly gain energy back)
        self.neuron_energy = torch.clamp(self.neuron_energy + 0.1, 0.0, 1.0)
            
        # 1. Drive initial target neurons (Focusing signals into Wernicke's Area)
        sensory_charge = torch.matmul(self.input_weights, sensory_inputs)
        sensory_charge[self.wernicke_end:] *= 0.1 # Dampen incoming sensory signal outside language sector
        
        # 2. Structural propagation across the internal brain
        synaptic_charge = torch.matmul(self.synaptic_matrix, current_state)
        
        # Combine charges and apply biological energy dampening (Exhausted cells can't fire heavily)
        total_potentials = torch.sigmoid(sensory_charge + synaptic_charge) * self.neuron_energy
        
        # Dynamic threshold shifted by emotions: Low Serotonin or High Noradrenaline heightens panic/sensitivity
        dynamic_threshold = self.base_threshold + (0.1 * (1.0 - self.neuron_serotonin.data))
        next_neuron_states = (total_potentials > dynamic_threshold).float()
        
        # Burn energy for neurons that successfully fired an action potential
        self.neuron_energy[next_neuron_states > 0] -= 0.4
        self.neuron_energy = torch.clamp(self.neuron_energy, 0.0, 1.0)
        
        # 3. Dynamic Hebbian Learning Loop (Plasticity)
        with torch.no_grad():
            learning_rate = 0.005
            plasticity_update = torch.outer(next_neuron_states, current_state)
            self.synaptic_matrix.add_(plasticity_update * learning_rate)
            self.synaptic_matrix.copy_(torch.clamp(self.synaptic_matrix, -1.0, 1.0))
        
        # 4. Computational Human Emotions Metrics
        happy_score = torch.sum(next_neuron_states * self.neuron_dopamine).item() / self.num_neurons
        sad_score = (1.0 - (torch.sum(next_neuron_states * self.neuron_serotonin).item() / self.num_neurons))
        angry_score = torch.sum(next_neuron_states * self.neuron_noradrenaline).item() / self.num_neurons
        
        emotions = {
            "Happy": min(max(happy_score * 3.0, 0.0), 1.0),
            "Sad": min(max(sad_score, 0.0), 1.0),
            "Angry": min(max(angry_score * 3.0, 0.0), 1.0)
        }
        
        # 5. Compile behavioral actions primarily pulling from Broca's Speech Area
        broca_mask = torch.zeros(self.num_neurons)
        broca_mask[self.broca_start:self.broca_end] = 1.0
        speech_layer = next_neuron_states * broca_mask
        
        actions = torch.matmul(self.output_weights, speech_layer)
        
        return actions, next_neuron_states, emotions

class PepeSchoolLifeEnvironment:
    """
    The school environment interface layer that maps Grade 9 classes, 
    the Room 204 dorm, and his intense emotional crush mappings into sensory data.
    """
    def __init__(self, model):
        self.model = model
        self.current_brain_state = None
        
        # Expanded vocabulary mapping academic concepts, locations, roommates, and romantic crush
        self.vocabulary = [
            "hello", "pepe", "human", "think", "feel", "remember",
            "school", "dorm", "study", "history", "english", "spanish",
            "grade9", "revolution", "poetry", "essay", "conjugation", "hola",
            "roommate", "leo", "max", "sam", "bunk", "bed", "talk",
            "crush", "like", "love", "heartbeat", "chloe", "blush", "nervous"
        ]
        
    def sentence_to_sensory_tensor(self, text):
        words = text.lower().strip().split()
        tensor = torch.zeros(self.model.inputs_dim)
        for i, word in enumerate(words[:self.model.inputs_dim]):
            if word in self.vocabulary:
                vocab_index = self.vocabulary.index(word)
                # Boost weight for critical emotional or educational terms
                weight = 2.0 if word in ["crush", "chloe", "history", "spanish", "english"] else 1.5
                tensor[vocab_index % self.model.inputs_dim] += weight
            else:
                tensor[hash(word) % self.model.inputs_dim] += 0.5
        return torch.nn.functional.normalize(tensor, dim=0)

    def data_signal_to_sentence(self, actions_tensor):
        top_values, top_indices = torch.topk(actions_tensor, 2)
        response_words = []
        for idx in top_indices:
            word_map = self.vocabulary[idx.item() % len(self.vocabulary)]
            response_words.append(word_map)
        return " ".join(response_words)

    def step(self, input_sentence):
        sensory_vector = self.sentence_to_sensory_tensor(input_sentence)
        actions, next_state, emotions = self.model(sensory_vector, current_state=self.current_brain_state)
        self.current_brain_state = next_state
        reply = self.data_signal_to_sentence(actions)
        return reply, emotions

if __name__ == "__main__":
    print("====================================================")
    print("🏫 PEPE: HIGH SCHOOL, DORM, & ROMANCE CONNECTOME V1.0")
    print("====================================================")
    
    # Instantiate Pepe's biological school blueprint
    pepe_brain = HumanScaleConnectome(num_neurons=2000, inputs_dim=35, outputs_dim=35)
    pepe_mind = PepeSchoolLifeEnvironment(pepe_brain)
    
    print("\n[System Boot] Pepe is awake inside Shared Dorm Room 204.")
    print("Roommates present: Leo (studying History), Max (sleeping), Sam (speaking Spanish).")
    print("Pepe is sitting at his desk. His secret crush is Chloe from Grade 9 English class.\n")
    
    # Running a school routine to trigger dynamic synaptic learning loops
    day_routine = [
        "school grade9 start history class american revolution",
        "english class write poetry essay rewrite memory",
        "spanish vocabulary study verb conjugation hola amigo",
        "return dorm room 204 sit bunk leo max sam talk",
        "see chloe walk past dorm window heartbeat crush blush nervous"
    ]
    
    for cycle in day_routine:
        print(f"📍 Event Action: {cycle}")
        reply, mood = pepe_mind.step(cycle)
        print(f"📊 Connectome Neurotransmitters -> Happy: {mood['Happy']:.1%}, Sad: {mood['Sad']:.1%}, Angry: {mood['Angry']:.1%}")
        print(f"🗣️ Pepe's Firing Thoughts: \"{reply}\"")
        print("-" * 60)
