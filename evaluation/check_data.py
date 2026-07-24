from pathlib import Path
import sys
import chromadb
from neo4j import GraphDatabase
from typing import Dict, List, Tuple
from collections import defaultdict
import statistics
root_dir = Path(__file__).parent.parent  # naik satu level dari evaluation ke root
sys.path.insert(0, str(root_dir))
from config import CONFIG

# Config
NEO4J_URI = CONFIG["neo4j_uri"]
NEO4J_USER = CONFIG["neo4j_user"]
NEO4J_PASSWORD = CONFIG["neo4j_password"]
CHROMA_PATH = CONFIG["chroma_path"]
CHROMA_COLLECTION = CONFIG["chroma_collection"]
RAW_COLLECTION = CONFIG["raw_collection"]

class DatabaseStats:
    """Class untuk mengumpulkan dan menampilkan statistik database."""
    
    def __init__(self):
        # Neo4j connection
        self.neo4j_driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        
        # ChromaDB connection
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.chroma_collection = self.chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION
        )
        self.chroma_raw_collection = self.chroma_client.get_or_create_collection(
            name=RAW_COLLECTION
        )
    
    def close(self):
        self.neo4j_driver.close()
    
    # ========================================================================
    # NEO4J STATISTICS
    # ========================================================================
    
    def get_neo4j_stats(self) -> Dict:
        """Mengambil semua statistik dari Neo4j."""
        stats = {}
        
        with self.neo4j_driver.session() as session:
            # Total nodes
            result = session.run("MATCH (n) RETURN count(n) as count")
            stats['total_nodes'] = result.single()['count']
            
            # Total Jurnal nodes
            result = session.run("MATCH (j:Jurnal) RETURN count(j) as count")
            stats['total_jurnal'] = result.single()['count']
            
            # Total Isi nodes (child)
            result = session.run("MATCH (i:Isi) RETURN count(i) as count")
            stats['total_isi'] = result.single()['count']
            
            # Total relationships
            result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            stats['total_relationships'] = result.single()['count']
            
            # HAS_SECTION relationships
            result = session.run("MATCH ()-[r:HAS_SECTION]->() RETURN count(r) as count")
            stats['has_section_rels'] = result.single()['count']
            
            # NEXT relationships
            result = session.run("MATCH ()-[r:NEXT]->() RETURN count(r) as count")
            stats['next_rels'] = result.single()['count']
            
            # Average chunks per jurnal
            result = session.run("""
                MATCH (j:Jurnal)-[:HAS_SECTION]->(i:Isi)
                WITH j, count(i) as chunk_count
                RETURN avg(chunk_count) as avg_chunks, 
                       min(chunk_count) as min_chunks,
                       max(chunk_count) as max_chunks,
                       percentileCont(chunk_count, 0.5) as median_chunks
            """).single()
            
            stats['avg_chunks_per_jurnal'] = result['avg_chunks'] or 0
            stats['min_chunks_per_jurnal'] = result['min_chunks'] or 0
            stats['max_chunks_per_jurnal'] = result['max_chunks'] or 0
            stats['median_chunks_per_jurnal'] = result['median_chunks'] or 0
            
            # Distribution of chunks per jurnal
            result = session.run("""
                MATCH (j:Jurnal)-[:HAS_SECTION]->(i:Isi)
                WITH j, count(i) as chunk_count
                RETURN chunk_count, count(*) as frequency
                ORDER BY chunk_count
            """)
            stats['chunk_distribution'] = [
                {"chunks": r['chunk_count'], "frequency": r['frequency']}
                for r in result
            ]
            
            # Top 5 jurnal with most chunks
            result = session.run("""
                MATCH (j:Jurnal)-[:HAS_SECTION]->(i:Isi)
                WITH j, count(i) as chunk_count
                RETURN j.judul as judul, chunk_count
                ORDER BY chunk_count DESC
                LIMIT 5
            """)
            stats['top_jurnal_chunks'] = [
                {"judul": r['judul'], "chunks": r['chunk_count']}
                for r in result
            ]
            
            # Sub_judul distribution (most common headings)
            result = session.run("""
                MATCH (i:Isi)
                WITH i.sub_judul as sub_judul, count(*) as count
                RETURN sub_judul, count
                ORDER BY count DESC
                LIMIT 10
            """)
            stats['top_sub_judul'] = [
                {"sub_judul": r['sub_judul'], "count": r['count']}
                for r in result
            ]
            
            # Average chunks per sub_judul
            result = session.run("""
                MATCH (i:Isi)
                WITH i.sub_judul as sub_judul, count(i) as chunk_count
                RETURN avg(chunk_count) as avg_chunks
            """).single()
            stats['avg_chunks_per_sub_judul'] = result['avg_chunks'] or 0
            
            # Total unique sub_judul
            result = session.run("""
                MATCH (i:Isi)
                RETURN count(distinct i.sub_judul) as unique_sub_judul
            """).single()
            stats['unique_sub_judul'] = result['unique_sub_judul'] or 0
            
            # Orphan Isi nodes (no HAS_SECTION relationship)
            result = session.run("""
                MATCH (i:Isi)
                WHERE NOT (i)<-[:HAS_SECTION]-()
                RETURN count(i) as orphan_count
            """).single()
            stats['orphan_isi_nodes'] = result['orphan_count'] or 0
            
            # Jurnal without any Isi nodes
            result = session.run("""
                MATCH (j:Jurnal)
                WHERE NOT (j)-[:HAS_SECTION]->()
                RETURN count(j) as empty_jurnal
            """).single()
            stats['empty_jurnal'] = result['empty_jurnal'] or 0
            
        return stats
    
    # ========================================================================
    # CHROMADB STATISTICS
    # ========================================================================
    
    def get_chroma_stats(self) -> Dict:
        """Mengambil statistik dari ChromaDB."""
        stats = {}
        
        # Improved collection
        stats['collection_name'] = CHROMA_COLLECTION
        stats['total_chunks'] = self.chroma_collection.count()
        
        # Get metadata stats
        try:
            # Count unique jurnal_id
            all_metadata = self.chroma_collection.get(include=['metadatas'])
            if all_metadata['metadatas']:
                jurnal_ids = set()
                for meta in all_metadata['metadatas']:
                    if meta and 'jurnal_id' in meta:
                        jurnal_ids.add(meta['jurnal_id'])
                
                stats['unique_jurnal_chroma'] = len(jurnal_ids)
                
                # Count chunks per jurnal
                jurnal_chunk_count = defaultdict(int)
                for meta in all_metadata['metadatas']:
                    if meta and 'jurnal_id' in meta:
                        jurnal_chunk_count[meta['jurnal_id']] += 1
                
                if jurnal_chunk_count:
                    chunks_per_jurnal = list(jurnal_chunk_count.values())
                    stats['avg_chunks_per_jurnal_chroma'] = statistics.mean(chunks_per_jurnal)
                    stats['min_chunks_per_jurnal_chroma'] = min(chunks_per_jurnal)
                    stats['max_chunks_per_jurnal_chroma'] = max(chunks_per_jurnal)
                    if len(chunks_per_jurnal) > 1:
                        stats['median_chunks_per_jurnal_chroma'] = statistics.median(chunks_per_jurnal)
                    else:
                        stats['median_chunks_per_jurnal_chroma'] = chunks_per_jurnal[0]
                else:
                    stats['avg_chunks_per_jurnal_chroma'] = 0
                    stats['min_chunks_per_jurnal_chroma'] = 0
                    stats['max_chunks_per_jurnal_chroma'] = 0
                    stats['median_chunks_per_jurnal_chroma'] = 0
        except Exception as e:
            print(f"Warning: Could not get Chroma metadata: {e}")
            stats['unique_jurnal_chroma'] = 0
            stats['avg_chunks_per_jurnal_chroma'] = 0
            stats['min_chunks_per_jurnal_chroma'] = 0
            stats['max_chunks_per_jurnal_chroma'] = 0
            stats['median_chunks_per_jurnal_chroma'] = 0
        
        # Raw collection
        try:
            stats['raw_collection_name'] = RAW_COLLECTION
            stats['total_raw_chunks'] = self.chroma_raw_collection.count()
            
            # Get raw metadata stats
            raw_metadata = self.chroma_raw_collection.get(include=['metadatas'])
            if raw_metadata['metadatas']:
                raw_jurnal_ids = set()
                for meta in raw_metadata['metadatas']:
                    if meta and 'jurnal_id' in meta:
                        raw_jurnal_ids.add(meta['jurnal_id'])
                stats['unique_jurnal_raw'] = len(raw_jurnal_ids)
                
                # Count raw chunks per jurnal
                raw_jurnal_count = defaultdict(int)
                for meta in raw_metadata['metadatas']:
                    if meta and 'jurnal_id' in meta:
                        raw_jurnal_count[meta['jurnal_id']] += 1
                
                if raw_jurnal_count:
                    raw_chunks_per_jurnal = list(raw_jurnal_count.values())
                    stats['avg_raw_chunks_per_jurnal'] = statistics.mean(raw_chunks_per_jurnal)
                    stats['min_raw_chunks_per_jurnal'] = min(raw_chunks_per_jurnal)
                    stats['max_raw_chunks_per_jurnal'] = max(raw_chunks_per_jurnal)
                    if len(raw_chunks_per_jurnal) > 1:
                        stats['median_raw_chunks_per_jurnal'] = statistics.median(raw_chunks_per_jurnal)
                    else:
                        stats['median_raw_chunks_per_jurnal'] = raw_chunks_per_jurnal[0]
                else:
                    stats['avg_raw_chunks_per_jurnal'] = 0
                    stats['min_raw_chunks_per_jurnal'] = 0
                    stats['max_raw_chunks_per_jurnal'] = 0
                    stats['median_raw_chunks_per_jurnal'] = 0
            else:
                stats['unique_jurnal_raw'] = 0
                stats['avg_raw_chunks_per_jurnal'] = 0
                stats['min_raw_chunks_per_jurnal'] = 0
                stats['max_raw_chunks_per_jurnal'] = 0
                stats['median_raw_chunks_per_jurnal'] = 0
                
        except Exception as e:
            print(f"Warning: Could not get raw Chroma metadata: {e}")
            stats['total_raw_chunks'] = 0
            stats['unique_jurnal_raw'] = 0
            stats['avg_raw_chunks_per_jurnal'] = 0
            stats['min_raw_chunks_per_jurnal'] = 0
            stats['max_raw_chunks_per_jurnal'] = 0
            stats['median_raw_chunks_per_jurnal'] = 0
        
        return stats
    
    # ========================================================================
    # DISPLAY METHODS
    # ========================================================================
    
    def display_neo4j_stats(self, stats: Dict):
        """Menampilkan statistik Neo4j dalam format yang rapi."""
        print("\n" + "="*70)
        print("📊 NEO4J STATISTICS")
        print("="*70)
        
        print(f"\n┌─ NODE TOTALS")
        print(f"│  • Total nodes (all labels)    : {stats['total_nodes']:,}")
        print(f"│  • Jurnal nodes (parent)       : {stats['total_jurnal']:,}")
        print(f"│  • Isi nodes (child)           : {stats['total_isi']:,}")
        print(f"│  • Orphan Isi nodes (no parent): {stats['orphan_isi_nodes']:,}")
        print(f"│  • Empty Jurnal (no child)     : {stats['empty_jurnal']:,}")
        print(f"└─")
        
        print(f"\n┌─ RELATIONSHIP TOTALS")
        print(f"│  • Total relationships         : {stats['total_relationships']:,}")
        print(f"│  • HAS_SECTION (parent→child)  : {stats['has_section_rels']:,}")
        print(f"│  • NEXT (child→child)          : {stats['next_rels']:,}")
        print(f"└─")
        
        print(f"\n┌─ CHUNK STATISTICS PER JURNAL")
        print(f"│  • Average chunks per jurnal   : {stats['avg_chunks_per_jurnal']:.2f}")
        print(f"│  • Median chunks per jurnal    : {stats['median_chunks_per_jurnal']:.2f}")
        print(f"│  • Min chunks per jurnal       : {stats['min_chunks_per_jurnal']}")
        print(f"│  • Max chunks per jurnal       : {stats['max_chunks_per_jurnal']}")
        print(f"└─")
        
        print(f"\n┌─ SUB-JUDUL STATISTICS")
        print(f"│  • Unique sub_judul            : {stats['unique_sub_judul']:,}")
        print(f"│  • Avg chunks per sub_judul    : {stats['avg_chunks_per_sub_judul']:.2f}")
        print(f"└─")
        
        # Distribution of chunks
        if stats['chunk_distribution']:
            print("\n┌─ CHUNK DISTRIBUTION PER JURNAL")
            for item in stats['chunk_distribution']:
                bar = "█" * min(item['frequency'], 40)
                print(f"│  {item['chunks']:>3} chunks : {item['frequency']:>3} jurnal {bar}")
            print(f"└─")
        
        # Top jurnal by chunks
        if stats['top_jurnal_chunks']:
            print("\n┌─ TOP 5 JURNAL BY CHUNK COUNT")
            for i, item in enumerate(stats['top_jurnal_chunks'], 1):
                judul = item['judul'][:50] + "..." if len(item['judul']) > 50 else item['judul']
                print(f"│  {i}. {judul:50} → {item['chunks']:>3} chunks")
            print(f"└─")
        
        # Top sub_judul
        if stats['top_sub_judul']:
            print("\n┌─ TOP 10 SUB-JUDUL (most frequent)")
            for i, item in enumerate(stats['top_sub_judul'], 1):
                sub = item['sub_judul'][:40] + "..." if len(item['sub_judul']) > 40 else item['sub_judul']
                print(f"│  {i:>2}. {sub:40} → {item['count']:>3} chunks")
            print(f"└─")
    
    def display_chroma_stats(self, stats: Dict):
        """Menampilkan statistik ChromaDB dalam format yang rapi."""
        print("\n" + "="*70)
        print("📊 CHROMADB STATISTICS")
        print("="*70)
        
        print(f"\n┌─ IMPROVED COLLECTION: {stats['collection_name']}")
        print(f"│  • Total chunks                : {stats['total_chunks']:,}")
        print(f"│  • Unique jurnal               : {stats['unique_jurnal_chroma']:,}")
        print(f"│  • Avg chunks per jurnal       : {stats['avg_chunks_per_jurnal_chroma']:.2f}")
        print(f"│  • Median chunks per jurnal    : {stats['median_chunks_per_jurnal_chroma']:.2f}")
        print(f"│  • Min chunks per jurnal       : {stats['min_chunks_per_jurnal_chroma']}")
        print(f"│  • Max chunks per jurnal       : {stats['max_chunks_per_jurnal_chroma']}")
        print(f"└─")
        
        print(f"\n┌─ RAW COLLECTION: {stats['raw_collection_name']}")
        print(f"│  • Total chunks                : {stats['total_raw_chunks']:,}")
        print(f"│  • Unique jurnal               : {stats['unique_jurnal_raw']:,}")
        print(f"│  • Avg chunks per jurnal       : {stats['avg_raw_chunks_per_jurnal']:.2f}")
        print(f"│  • Median chunks per jurnal    : {stats['median_raw_chunks_per_jurnal']:.2f}")
        print(f"│  • Min chunks per jurnal       : {stats['min_raw_chunks_per_jurnal']}")
        print(f"│  • Max chunks per jurnal       : {stats['max_raw_chunks_per_jurnal']}")
        print(f"└─")
    
    def display_summary(self, neo4j_stats: Dict, chroma_stats: Dict):
        """Menampilkan ringkasan perbandingan antara Neo4j dan ChromaDB."""
        print("\n" + "="*70)
        print("📈 SUMMARY & COMPARISON")
        print("="*70)
        
        print(f"\n┌─ NODE COMPARISON")
        print(f"│  Neo4j Jurnal (parent nodes)  : {neo4j_stats['total_jurnal']:,}")
        print(f"│  Neo4j Isi (child nodes)      : {neo4j_stats['total_isi']:,}")
        print(f"│  ChromaDB chunks (improved)   : {chroma_stats['total_chunks']:,}")
        print(f"│  ChromaDB chunks (raw)        : {chroma_stats['total_raw_chunks']:,}")
        print(f"└─")
        
        # Parent-child ratio
        if neo4j_stats['total_jurnal'] > 0:
            child_per_parent = neo4j_stats['total_isi'] / neo4j_stats['total_jurnal']
            print(f"\n┌─ PARENT-CHILD RATIO")
            print(f"│  Average Isi nodes per Jurnal : {child_per_parent:.2f}")
            print(f"│  (In Neo4j graph)")
            print(f"└─")
        
        # Chroma comparison
        if chroma_stats['total_chunks'] > 0 and chroma_stats['total_raw_chunks'] > 0:
            diff = chroma_stats['total_chunks'] - chroma_stats['total_raw_chunks']
            pct = (diff / chroma_stats['total_raw_chunks'] * 100) if chroma_stats['total_raw_chunks'] > 0 else 0
            print(f"\n┌─ IMPROVED vs RAW")
            print(f"│  Improved chunks - Raw chunks : {diff:+,}")
            print(f"│  Difference percentage        : {pct:+.1f}%")
            print(f"│  (Improved usually has fewer chunks due to section merging)")
            print(f"└─")
    
    def run(self):
        """Menjalankan semua statistik dan menampilkan hasilnya."""
        try:
            print("\n" + "🔍"*35)
            print("   DATABASE STATISTICS ANALYZER")
            print("🔍"*35)
            
            print("\n⏳ Mengambil statistik dari Neo4j...")
            neo4j_stats = self.get_neo4j_stats()
            self.display_neo4j_stats(neo4j_stats)
            
            print("\n⏳ Mengambil statistik dari ChromaDB...")
            chroma_stats = self.get_chroma_stats()
            self.display_chroma_stats(chroma_stats)
            
            self.display_summary(neo4j_stats, chroma_stats)
            
            print("\n" + "="*70)
            print("✅ Selesai!")
            print("="*70)
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.close()


# ========================================================================
# MAIN
# ========================================================================

def main():
    stats = DatabaseStats()
    stats.run()

if __name__ == "__main__":
    main()