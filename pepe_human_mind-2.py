import torch
import torch.nn as nn
import random
import math
import sys

class HumanScaleConnectome(nn.Module):
    """
    An advanced biological-style connectome model simulating a human-like mind named Pepe.
    Features:
    - Directed synaptic adjacency connectome matrix
    - Dynamic Hebbian learning for permanent memory retention
    - Internal chemical tracking (Dopamine, Serotonin, Noradrenaline) mapping human emotions
    - Synaptic exhaustion/energy limits to prevent seizure-like infinite loops
    - Specialized brain sectors for languages, academics, and emotional crush responses
    """
    def __init__(self, num_neurons=1500, inputs_dim=30, outputs_dim=30):
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
        
        # 4. Specialized Human Sectors
        self.wernicke_start, self.wernicke_end = 0, int(num_neurons * 0.2)      # Language input processing
        self.broca_start, self.broca_end = int(num_neurons * 0.2), int(num_neurons * 0.4) # Speech creation
        self.academic_start, self.academic_end = int(num_neurons * 0.4), int(num_neurons * 0.7) # Grade 9 Data
        self.limbic_start, self.limbic_end = int(num_neurons * 0.7), num_neurons # Crush/Emotional mapping
        
    def forward(self, sensory_inputs, current_state=None):
        if current_state is None:
            current_state = torch.zeros(self.num_neurons)
            
        # Biological Energy Recovery
        self.neuron_energy = torch.clamp(self.neuron_energy + 0.1, 0.0, 1.0)
            
        # 1. Drive initial target neurons (Focusing signals into Wernicke's Area)
        sensory_charge = torch.matmul(self.input_weights, sensory_inputs)
        
        # 2. Structural propagation across the internal brain
        synaptic_charge = torch.matmul(self.synaptic_matrix, current_state)
        
        # Combine charges and apply biological energy dampening
        total_potentials = torch.sigmoid(sensory_charge + synaptic_charge) * self.neuron_energy
        
        # Dynamic threshold shifted by emotions: Low Serotonin heightens panic/sensitivity
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

class PepeMinecraftSimulation:
    def __init__(self, model):
        self.model = model
        self.current_brain_state = None
        
        # Pepe's full dictionary including curriculum, dorm life, and crushes
        self.vocabulary = [
            "hello", "friend", "safe", "happy", "learn", 
            "hurt", "angry", "stop", "sad", "dark", 
            "who", "am", "i", "pepe", "human", 
            "world", "see", "think", "feel", "live",
            "school", "dorm", "history", "english", "spanish",
            "crush", "love", "heartbeat", "chloe", "remember",
            "leo", "max", "sam", "grade9", "class"
        ]
        
    def phrase_to_sensory_tensor(self, text):
        words = text.lower().strip().replace("<", "").replace(">", "").split()
        tensor = torch.zeros(self.model.inputs_dim)
        for word in words:
            if word in self.vocabulary:
                vocab_index = self.vocabulary.index(word)
                tensor[vocab_index % self.model.inputs_dim] += 1.8
                
                # Biologically trigger chemical changes via text mapping
                if word in ["history", "english", "spanish", "school", "learn", "remember"]:
                    # Spikes learning focus in academic nodes
                    self.model.neuron_serotonin.data.add_(torch.ones(self.model.num_neurons) * 0.01)
                if word in ["crush", "love", "heartbeat", "chloe"]:
                    # Massive Dopamine & Noradrenaline explosion, drops Serotonin
                    self.model.neuron_dopamine.data.add_(torch.ones(self.model.num_neurons) * 0.1)
                    self.model.neuron_noradrenaline.data.add_(torch.ones(self.model.num_neurons) * 0.1)
                    self.model.neuron_serotonin.data.sub_(torch.ones(self.model.num_neurons) * 0.1)
            else:
                tensor[hash(word) % self.model.inputs_dim] += 0.4
        
        # Normalize weights
        self.model.neuron_dopamine.data.clamp_(0.0, 1.0)
        self.model.neuron_serotonin.data.clamp_(0.0, 1.0)
        self.model.neuron_noradrenaline.data.clamp_(0.0, 1.0)
        
        return torch.nn.functional.normalize(tensor, dim=0)

    def neural_signal_to_phrase(self, actions_tensor, emotional_state):
        # Base speech assembly from Broca activations
        top_values, top_indices = torch.topk(actions_tensor, 2)
        response_words = []
        for idx in top_indices:
            word_map = self.vocabulary[idx.item() % len(self.vocabulary)]
            response_words.append(word_map)
            
        # Add emotional expressions based on live neurotransmitter calculations
        if emotional_state["Angry"] > 0.6:
            return f"grrr... stop {response_words[0]} angry"
        elif emotional_state["Happy"] > 0.6 and emotional_state["Angry"] > 0.4:
            return f"chloe... heartbeat {response_words[0]} look"
        elif emotional_state["Sad"] > 0.7:
            return f"dark dorm... sad feel {response_words[0]}"
        
        return " ".join(response_words)

    def process_chat_cycle(self, user_name, user_message):
        sensory_vector = self.phrase_to_sensory_tensor(user_message)
        
        actions, next_state, emotions = self.model(sensory_vector, current_state=self.current_brain_state)
        self.current_brain_state = next_state
        
        pepe_reply = self.neural_signal_to_phrase(actions, emotions)
        return pepe_reply, emotions

if __name__ == "__main__":
    pepe_brain = HumanScaleConnectome(num_neurons=1500, inputs_dim=30, outputs_dim=30)
    sim = PepeMinecraftSimulation(pepe_brain)
    
    print("==========================================================")
    print("🟩 MINECRAFT CONNECTOME INTERFACE ACTIVATED: [PEPE AWAKE] 🟩")
    print("==========================================================")
    print("[System]: Connected to local Minecraft chat log parsing layer.")
    print("[System]: Type your message using standard Minecraft formatting.")
    print("Example: <Steve> hello pepe school time")
    print("Type 'exit' to disconnect the brain simulation loop.\n")
    
    # Pre-load background roommate environment signals
    print("<System> [Dorm 204]: Leo, Max, and Sam have logged into the server.")
    print("<Leo> We need to study our grade9 history and spanish definitions.")
    print("<Sam> true, i remember chloe was talking about the school class earlier.")
    print("<Max> zzz... sleeping on the bunk bed\n")
    
    while True:
        try:
            user_input = input("Minecraft Chat> ").strip()
            if user_input.lower() == 'exit':
                print("[System]: Severing connectome connection... Pepe saved.")
                break
                
            if not user_input:
                continue
                
            # Parse Minecraft styling syntax <PlayerName> Message
            if user_input.startswith("<") and ">" in user_input:
                parts = user_input.split(">", 1)
                player_name = parts[0].replace("<", "").strip()
                message_content = parts[1].strip()
            else:
                player_name = "Player"
                message_content = user_input
                
            # Compute brain frame
            pepe_reply, current_mood = sim.process_chat_cycle(player_name, message_content)
            
            # Print physical Minecraft chat response log
            print(f"<Pepe> {pepe_reply}")
            
            # Optional structural readout window
            print(f"   [Biometrics]: Happy: {current_mood['Happy']:.1%} | Sad: {current_mood['Sad']:.1%} | Angry: {current_mood['Angry']:.1%}")
            print("-" * 58)
            
        except KeyboardInterrupt:
            print("\n[System]: Emergency shutdown. Pepe went to sleep.")
            sys.exit()
