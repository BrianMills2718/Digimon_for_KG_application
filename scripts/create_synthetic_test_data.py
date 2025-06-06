#!/usr/bin/env python3
"""Create synthetic test data for demonstrating DIGIMON capabilities"""

import os
import json

def create_synthetic_corpus():
    """Create a synthetic corpus with diverse content for testing all capabilities"""
    
    # Create directory
    os.makedirs("Data/Synthetic_Test", exist_ok=True)
    
    # Document 1: Technology and AI
    doc1 = """Title: The Rise of Artificial Intelligence in Modern Society

Artificial Intelligence (AI) has rapidly evolved from a concept in science fiction to a fundamental technology shaping our daily lives. Major tech companies like Google, Microsoft, and OpenAI are leading the charge in developing advanced AI systems. 

Dr. Sarah Chen, a prominent AI researcher at Stanford University, recently stated that "We are witnessing an unprecedented acceleration in AI capabilities." Her research team has developed new neural network architectures that can process information more efficiently than ever before.

The relationship between AI and society is complex. While AI brings numerous benefits such as improved healthcare diagnostics and personalized education, it also raises concerns about job displacement and privacy. Tech entrepreneur John Martinez founded EthicalAI Corp specifically to address these challenges.

In healthcare, AI systems developed by companies like DeepMind have shown remarkable success. The collaboration between DeepMind and hospitals in London has resulted in AI that can detect eye diseases with accuracy matching human specialists. Dr. Emily Watson, who leads this initiative, believes this is just the beginning.

The economic impact is significant. According to economist Prof. Robert Kim, AI could contribute $15 trillion to the global economy by 2030. However, labor union leader Maria Rodriguez warns that "We must ensure that AI benefits all workers, not just shareholders."

Education is another frontier. AI tutoring systems developed by Prof. David Liu at MIT can adapt to individual learning styles. His startup, LearnAI, has partnered with schools across Massachusetts to implement personalized learning programs."""

    # Document 2: Climate Change and Environment
    doc2 = """Title: Global Climate Action and Environmental Challenges

Climate change remains one of the most pressing challenges of our time. The recent IPCC report, led by climate scientist Dr. Michael Thompson, warns that we have less than a decade to make substantial changes to avoid catastrophic warming.

Environmental activist Greta Andersen has mobilized millions of young people worldwide through her organization, Youth4Climate. She recently met with President Emma Williams to discuss new climate policies. The president announced ambitious targets to reduce carbon emissions by 50% by 2030.

The relationship between industry and environment is evolving. CEO James Park of GreenTech Industries has pioneered sustainable manufacturing processes. His company collaborates with environmental NGOs like the Earth Foundation, headed by Dr. Lisa Chang, to ensure ecological responsibility.

Renewable energy is booming. Solar panel manufacturer SunPower Corp, under the leadership of CEO Rachel Green, has developed new photovoltaic technology that increases efficiency by 40%. Wind energy expert Prof. Carlos Martinez at the University of Barcelona predicts that "renewable sources will dominate the energy mix within 20 years."

The fossil fuel industry faces transformation. Oil executive Thomas Anderson shocked shareholders by announcing PetroGlobal's pivot to renewable energy. This decision followed pressure from investor coalition leader Sophie Laurent, who manages $2 trillion in climate-conscious funds.

Cities are taking action. Mayor Jennifer Kim of San Francisco has implemented the most aggressive urban climate plan in the US. Her administration works closely with urban planner Dr. Ahmad Hassan to create carbon-neutral neighborhoods."""

    # Document 3: Politics and Society
    doc3 = """Title: Political Polarization and Social Media in the Digital Age

The intersection of politics and social media has fundamentally altered democratic discourse. Senator Patricia Johnson, chair of the Technology Committee, has proposed new regulations for social media platforms. Her bill is co-sponsored by Senator Mark Davis, despite their different party affiliations.

Media researcher Dr. Angela Martinez at Columbia University studies how algorithms shape political views. Her research shows that "echo chambers on social media are intensifying political polarization." This finding has prompted Facebook CEO Kevin Liu to announce changes to the platform's recommendation system.

The relationship between traditional media and new media is complex. Veteran journalist Bob Smith of the Washington Herald argues that "social media has democratized information but also weaponized misinformation." His colleague, investigative reporter Jane Chen, recently exposed a network of bot accounts influencing elections.

Political consultant Tom Williams has adapted to the digital age by founding DataPolitics Inc. His firm uses AI to analyze voter sentiment, working with candidates from both major parties. Ethics professor Dr. Susan Brown warns about the implications of micro-targeting in political campaigns.

Grassroots movements have found new power online. Activist leader Marcus Washington organized nationwide protests through social media, bringing together diverse coalitions. His ally, labor organizer Elena Rodriguez, used similar tactics to coordinate strikes across multiple industries.

The role of fact-checking has become crucial. FactCheck.org director Dr. Richard Park has expanded operations to combat misinformation. He collaborates with tech platforms and news organizations, including a partnership with CNN anchor Lisa Anderson."""

    # Document 4: Space Exploration and Science
    doc4 = """Title: The New Space Race and Scientific Discoveries

Space exploration has entered a new golden age with both government agencies and private companies pushing boundaries. NASA Administrator Dr. James Chen announced plans for a permanent lunar base by 2035. This ambitious project involves collaboration with SpaceX, led by visionary entrepreneur Elena Musk.

The relationship between NASA and private space companies has evolved from competition to cooperation. Blue Origin founder Jeffrey Bezos recently met with Dr. Chen to discuss joint missions to Mars. European Space Agency director Dr. Marie Dubois emphasizes that "international cooperation is essential for humanity's future in space."

Scientific breakthroughs continue to amaze. Astrophysicist Dr. Raj Patel at Caltech discovered a potentially habitable exoplanet using the James Webb Space Telescope. His colleague, Dr. Sarah Kim, developed new techniques for analyzing atmospheric composition of distant worlds.

The commercial space industry is booming. Satellite company StarLink, headed by CEO Michael Torres, provides internet to remote areas worldwide. Competitor OneWeb, led by CEO Amanda Singh, focuses on connecting underserved communities in developing nations.

Space tourism is becoming reality. Virgin Galactic pilot Captain John Anderson completed his 100th suborbital flight, carrying researchers and tourists. Space hotel architect Dr. Yuki Tanaka designs habitats for extended stays in orbit, working with construction firm Orbital Builders.

The search for extraterrestrial life intensifies. SETI researcher Dr. Carlos Mendez detected unusual signals from the Alpha Centauri system. Astrobiologist Dr. Emma Wilson studies extremophiles on Earth to understand potential alien life forms. Their work is funded by billionaire philanthropist Robert Sterling."""

    # Write documents
    with open("Data/Synthetic_Test/technology_ai.txt", "w") as f:
        f.write(doc1)
    
    with open("Data/Synthetic_Test/climate_environment.txt", "w") as f:
        f.write(doc2)
    
    with open("Data/Synthetic_Test/politics_society.txt", "w") as f:
        f.write(doc3)
    
    with open("Data/Synthetic_Test/space_science.txt", "w") as f:
        f.write(doc4)
    
    print("✓ Created synthetic corpus with 4 documents in Data/Synthetic_Test/")
    print("  - technology_ai.txt")
    print("  - climate_environment.txt") 
    print("  - politics_society.txt")
    print("  - space_science.txt")

def create_mini_discourse_corpus():
    """Create a mini corpus specifically for discourse analysis testing"""
    
    os.makedirs("Data/Discourse_Test", exist_ok=True)
    
    # Discourse document with clear narrative patterns
    discourse_doc = """Title: The Great Debate on Universal Basic Income

PROPONENTS' NARRATIVE:
Tech billionaire Andrew Yang champions Universal Basic Income (UBI) as "the solution to automation-driven unemployment." He frames it as "investing in human potential" and uses emotional appeals: "No one should lose dignity because a robot took their job."

Economist Dr. Sarah Mitchell supports UBI with data: "Our studies show UBI reduces poverty by 40% without decreasing work participation." She positions critics as "clinging to outdated economic models."

OPPONENTS' COUNTER-NARRATIVE:  
Senator Bob Thompson attacks UBI as "socialist welfare that will bankrupt America." He uses fear-based rhetoric: "Do you want your tax dollars funding laziness?" His framing emphasizes personal responsibility.

Business leader Karen White argues "UBI will destroy work ethic and innovation." She shares anecdotes of "welfare dependency" and warns of "creating a permanent underclass."

RHETORICAL STRATEGIES:
- Proponents use: Future-oriented language, appeals to compassion, economic data
- Opponents use: Traditional values, fear of change, individual responsibility  

DISCOURSE PATTERNS:
1. Competing definitions: "Safety net" vs "Handout"
2. Metaphorical framing: "Foundation for growth" vs "Trap of dependency"  
3. Selective evidence: Both sides cherry-pick studies
4. Identity politics: "Innovators" vs "Hard workers"

The media amplifies polarization. CNN's Jane Foster presents UBI favorably while Fox's Mike Johnson warns of "radical socialism." Social media echo chambers reinforce existing beliefs."""

    with open("Data/Discourse_Test/ubi_debate.txt", "w") as f:
        f.write(discourse_doc)
    
    print("\n✓ Created discourse analysis test corpus in Data/Discourse_Test/")
    print("  - ubi_debate.txt")

def create_community_graph_corpus():
    """Create a corpus designed to have clear community structures"""
    
    os.makedirs("Data/Community_Test", exist_ok=True)
    
    # Document with clear community clusters
    community_doc = """Title: The Tech Startup Ecosystem of Silicon Valley

COMMUNITY 1: AI/ML Startups
OpenAI, led by Sam Altman, collaborates closely with Anthropic (Dario Amodei) and Cohere (Aidan Gomez). These companies share researchers, with Dr. Ilya Sutskever mentoring across organizations. They compete but also co-publish papers. Investment firm Andreessen Horowitz funds all three.

DeepMind (Demis Hassabis) maintains connections through former employees who joined Inflection AI (Mustafa Suleyman). Research scientist Dr. Fei-Fei Li advises multiple companies in this cluster.

COMMUNITY 2: Fintech Cluster  
Stripe (Patrick Collison) has deep ties with Square (Jack Dorsey) through shared payment infrastructure. Plaid (Zach Perret) connects both to banks. PayPal mafia members including Peter Thiel influence this entire ecosystem.

Robinhood (Vlad Tenev) and Coinbase (Brian Armstrong) share regulatory consultants. Both work with the same law firm, Wilson Sonsini. Venture capitalist Marc Andreessen sits on multiple boards.

COMMUNITY 3: Social Media Innovators
Instagram founders Kevin Systrom and Mike Krieger maintain connections with WhatsApp's Jan Koum. All have ties to Facebook/Meta through acquisition. TikTok's Shou Zi Chew recruited former Instagram employees.

Twitter/X under Elon Musk hired engineers from Meta. LinkedIn (Reid Hoffman) serves as the professional network connecting all players.

INTER-COMMUNITY BRIDGES:
- Investor Sequoia Capital funds companies across all communities  
- Stanford University professors advise startups in each cluster
- Y Combinator alumni network spans all sectors

These communities compete for talent, creating a dynamic ecosystem where employees move between clusters, carrying knowledge and connections."""

    with open("Data/Community_Test/startup_ecosystem.txt", "w") as f:
        f.write(community_doc)
    
    print("\n✓ Created community detection test corpus in Data/Community_Test/")
    print("  - startup_ecosystem.txt")

if __name__ == "__main__":
    print("Creating synthetic test data for DIGIMON capabilities...")
    print("=" * 60)
    
    create_synthetic_corpus()
    create_mini_discourse_corpus()
    create_community_graph_corpus()
    
    print("\n✓ All synthetic test data created successfully!")
    print("\nYou can now test DIGIMON with these datasets:")
    print("  - Synthetic_Test: General purpose testing (4 documents)")
    print("  - Discourse_Test: Discourse analysis testing")
    print("  - Community_Test: Community detection testing")