#!/usr/bin/env python3
"""
DIGIMON GraphRAG CLI

A command-line interface for querying documents using the DIGIMON GraphRAG pipeline.
This CLI orchestrates the full pipeline from corpus preparation to answer generation.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from pyfiglet import Figlet
from colorama import Fore, Style, init

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from Core.Common.Logger import logger
from Core.Common.ContextMixin import ContextMixin
from Core.AgentBrain.agent_brain import PlanningAgent
from Core.AgentOrchestrator.orchestrator import AgentOrchestrator
from Core.AgentSchema.context import GraphRAGContext
from Option.Config2 import Config
from Core.Provider.LiteLLMProvider import LiteLLMProvider
from Core.Index.EmbeddingFactory import get_rag_embedding
from Core.Chunk.ChunkFactory import ChunkFactory

# Initialize colorama
init(autoreset=True)


class DigimonCLI:
    """Main CLI class for DIGIMON GraphRAG system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the CLI with optional configuration.
        
        Args:
            config_path: Path to configuration file (defaults to Option/Config2.yaml)
        """
        self.config_path = config_path or "Option/Config2.yaml"
        self.context: Optional[GraphRAGContext] = None
        self.orchestrator: Optional[AgentOrchestrator] = None
        self.planning_agent: Optional[PlanningAgent] = None
        self.interactive_mode = False
        self.corpus_path = None
        self.react_mode = False
        
    def print_welcome(self):
        """Print the welcome banner."""
        f = Figlet(font='big')
        logo = f.renderText('DIGIMON')
        print(f"{Fore.GREEN}{'#' * 100}{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}{logo}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}DIGIMON GraphRAG CLI - Natural Language Document Querying{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}\n")
        
    async def initialize_system(self):
        """Initialize the GraphRAG system components."""
        try:
            logger.info("Initializing DIGIMON GraphRAG system...")
            
            # Load configuration
            if self.config_path and os.path.exists(self.config_path):
                config = Config.from_yaml_file(self.config_path)
            else:
                config = Config.default()
            
            # Initialize providers
            llm_instance = LiteLLMProvider(config.llm)
            encoder_instance = get_rag_embedding(config=config)
            chunk_factory = ChunkFactory(config)
            
            # Create context with all required fields
            self.context = GraphRAGContext(
                target_dataset_name="cli_dataset",  # Default dataset name for CLI
                main_config=config,
                llm_provider=llm_instance,
                embedding_provider=encoder_instance,
                chunk_storage_manager=chunk_factory
            )
            
            # Initialize orchestrator
            self.orchestrator = AgentOrchestrator(
                main_config=config,
                llm_instance=llm_instance,
                encoder_instance=encoder_instance,
                chunk_factory=chunk_factory,
                graphrag_context=self.context
            )
            
            # Initialize planning agent
            self.planning_agent = PlanningAgent(
                config=config,
                graphrag_context=self.context
            )
            
            logger.info("System initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            return False
            
    async def prepare_corpus(self, corpus_path: str):
        """
        Prepare the corpus for processing.
        
        Args:
            corpus_path: Path to the directory containing documents
        """
        logger.info(f"Preparing corpus from: {corpus_path}")
        
        # The planning agent will handle corpus preparation through its tools
        # We just need to ensure the path exists
        if not os.path.exists(corpus_path):
            raise ValueError(f"Corpus path does not exist: {corpus_path}")
            
        self.corpus_path = corpus_path
            
        return True
        
    async def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a single query through the GraphRAG pipeline.
        
        Args:
            query: Natural language query
            
        Returns:
            Dictionary containing the answer and context
        """
        logger.info(f"Processing query: {query}")
        
        try:
            # Extract corpus name from path
            corpus_name = os.path.basename(self.corpus_path) if self.corpus_path else "DefaultDataset"
            
            # Process query through planning agent with corpus information
            if self.react_mode:
                logger.info("Using ReAct-style processing")
                result = await self.planning_agent.process_query_react(
                    query,
                    actual_corpus_name=corpus_name
                )
            else:
                result = await self.planning_agent.process_query(
                    query,
                    actual_corpus_name=corpus_name
                )
            
            if isinstance(result, dict) and result.get("error"):
                logger.error(f"Query processing failed: {result.get('error')}")
                return result
                
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "error": str(e),
                "generated_answer": f"Error processing query: {str(e)}"
            }
            
    def format_result(self, result: Dict[str, Any]):
        """
        Format and display the query result.
        
        Args:
            result: Query result dictionary
        """
        print(f"\n{Fore.GREEN}{'=' * 60}{Style.RESET_ALL}")
        
        if result.get("error"):
            print(f"{Fore.RED}Error: {result['error']}{Style.RESET_ALL}")
        
        answer = result.get("generated_answer", "No answer generated")
        print(f"{Fore.CYAN}Answer:{Style.RESET_ALL}\n{answer}")
        
        # Optionally display context details
        if result.get("retrieved_context") and not result.get("error"):
            print(f"\n{Fore.YELLOW}Context Retrieved:{Style.RESET_ALL}")
            # Show a summary of retrieved entities/chunks
            context = result["retrieved_context"]
            if isinstance(context, dict):
                for key, value in context.items():
                    if key == "Entity.VDBSearch" and isinstance(value, dict):
                        entities = value.get("search_results", [])
                        if entities:
                            print(f"  - Found {len(entities)} relevant entities")
                    elif key == "Chunk.GetTextForEntities" and isinstance(value, dict):
                        chunks = value.get("retrieved_chunks", [])
                        if chunks:
                            print(f"  - Retrieved {len(chunks)} text chunks")
                            
        print(f"{Fore.GREEN}{'=' * 60}{Style.RESET_ALL}\n")
        
    async def run_interactive_mode(self):
        """Run the CLI in interactive mode."""
        self.interactive_mode = True
        
        print(f"{Fore.YELLOW}Entering interactive mode. Type 'quit' or 'exit' to leave.{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Type 'help' for available commands.{Style.RESET_ALL}\n")
        
        while True:
            try:
                # Get user input
                query = input(f"{Fore.GREEN}digimon> {Style.RESET_ALL}").strip()
                
                # Handle special commands
                if query.lower() in ['quit', 'exit']:
                    print(f"{Fore.YELLOW}Exiting interactive mode...{Style.RESET_ALL}")
                    break
                elif query.lower() == 'help':
                    self.print_help()
                    continue
                elif query.lower() == 'clear':
                    os.system('clear' if os.name == 'posix' else 'cls')
                    continue
                elif not query:
                    continue
                    
                # Process the query
                result = await self.process_query(query)
                self.format_result(result)
                
            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}Use 'quit' or 'exit' to leave interactive mode.{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
                
    def print_help(self):
        """Print help information for interactive mode."""
        help_text = """
Available commands:
  - Type any natural language query to search the corpus
  - 'help'  - Show this help message
  - 'clear' - Clear the screen
  - 'quit'  - Exit interactive mode
  - 'exit'  - Exit interactive mode
  
Examples:
  - "What are the main concepts discussed in the documents?"
  - "Find information about machine learning algorithms"
  - "Summarize the key findings related to neural networks"
        """
        print(f"{Fore.CYAN}{help_text}{Style.RESET_ALL}")
        
    async def run_batch_mode(self, queries_file: str, output_file: Optional[str] = None):
        """
        Run the CLI in batch mode with queries from a file.
        
        Args:
            queries_file: Path to file containing queries (one per line)
            output_file: Optional path to save results
        """
        logger.info(f"Running batch mode with queries from: {queries_file}")
        
        try:
            with open(queries_file, 'r') as f:
                queries = [line.strip() for line in f if line.strip()]
                
            results = []
            
            for i, query in enumerate(queries, 1):
                print(f"\n{Fore.CYAN}Processing query {i}/{len(queries)}: {query}{Style.RESET_ALL}")
                result = await self.process_query(query)
                results.append({
                    "query": query,
                    "result": result
                })
                self.format_result(result)
                
            # Save results if output file specified
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Results saved to: {output_file}")
                
        except Exception as e:
            logger.error(f"Error in batch mode: {e}")
            

async def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="DIGIMON GraphRAG CLI - Natural Language Document Querying",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode with default corpus
  python digimon_cli.py -i -c /path/to/corpus

  # Single query mode
  python digimon_cli.py -c /path/to/corpus -q "What are the main topics?"

  # Batch mode
  python digimon_cli.py -c /path/to/corpus -b queries.txt -o results.json

  # Use custom config
  python digimon_cli.py -c /path/to/corpus -i --config custom_config.yaml
        """
    )
    
    parser.add_argument(
        '-c', '--corpus',
        type=str,
        required=True,
        help='Path to the corpus directory containing documents'
    )
    
    parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    
    parser.add_argument(
        '-q', '--query',
        type=str,
        help='Single query to process (non-interactive mode)'
    )
    
    parser.add_argument(
        '-b', '--batch',
        type=str,
        help='Batch mode: Process queries from file'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output file for batch results (JSON format)'
    )
    
    parser.add_argument(
        '--react', action='store_true',
        help='Use experimental ReAct-style iterative planning (plan → act → observe → repeat)'
    )
    
    parser.add_argument(
        '--config', type=str, default='Option/Config2.yaml',
        help='Path to configuration file (default: Option/Config2.yaml)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not (args.interactive or args.query or args.batch):
        parser.error("Must specify one of: --interactive, --query, or --batch")
        
    if args.batch and args.interactive:
        parser.error("Cannot use --batch and --interactive together")
        
    # Initialize CLI
    cli = DigimonCLI(config_path=args.config)
    
    # Print welcome banner
    cli.print_welcome()
    
    # Set react mode
    cli.react_mode = args.react
    if cli.react_mode:
        print(f"{Fore.YELLOW}ReAct mode enabled - using iterative planning{Style.RESET_ALL}")
        
    # Initialize system
    if not await cli.initialize_system():
        print(f"{Fore.RED}Failed to initialize system. Exiting.{Style.RESET_ALL}")
        return 1
        
    # Prepare corpus
    try:
        await cli.prepare_corpus(args.corpus)
    except Exception as e:
        print(f"{Fore.RED}Failed to prepare corpus: {e}{Style.RESET_ALL}")
        return 1
        
    # Run appropriate mode
    try:
        if args.interactive:
            await cli.run_interactive_mode()
        elif args.query:
            result = await cli.process_query(args.query)
            cli.format_result(result)
        elif args.batch:
            await cli.run_batch_mode(args.batch, args.output)
            
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    # Run the async main function
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
