from tempfile import NamedTemporaryFile
import os

from sqlalchemy.exc import ProgrammingError
from sqlalchemy import text
from pytest import fixture, raises

from src.data_gent.connection import get_sqlalchemy_engine
from src.data_gent.settings import settings
from src.data_gent.load import load_document
from src.data_gent.retrieval import retrieve, RetrievalResult, vector_search, fts_search
from src.data_gent.embeddings import TestEmbeddingSource
from src.data_gent.chunking import SemchunkChunker


@fixture
def docfile():
    data = """
"Big River" is a song written and originally recorded by Johnny Cash. Released as a single by Sun Records in 1958, it went as high as #4 on the Billboard country music charts and stayed on the charts for 14 weeks.[2] The song tells a story of the chase of a lost love along the course of Mississippi River from Saint Paul, Minnesota to New Orleans, Louisiana.
Background

A verse omitted from the original recording was later performed during Johnny Cash's live performances.[3] A demo recording from the Sun sessions featuring the omitted verse also exists and has been released on numerous Sun compilations.
Chart performance
Chart (1958) 	Peak
position
U.S. Billboard Hot Country Singles 	4
U.S. Billboard Hot 100 	14
Cover versions

    Ian Tyson (of Ian and Sylvia) included a spirited version of Big River on the duo's Lovin' Sound album released in 1967, with David Rae on lead guitar.
    The Grateful Dead played a cover version of this song 396 times from 1965-1995.[4] First appearing on their 1976 live album Steal Your Face, it features on many of their concert recordings, such as One from the Vault and Dick's Picks Volume 1.
    Colin Linden recorded a version included in the 2003 tribute album, Johnny's Blues: A Tribute To Johnny Cash (Northern Blues).
    Trick Pony recorded a version of Big River with Johnny Cash and Waylon Jennings on their debut album.
    Hank Williams Jr. covered this song on his 1970 album Singing My Songs - Johnny Cash, which contained exclusively covers of Johnny Cash songs.
    The Secret Sisters recorded a version of the song in 2011, with Jack White playing backing guitar.[5]
    Bob Dylan and The Band recorded two takes of the song in 1967 during The Basement Tapes sessions. They were officially released on November 4, 2014, on The Bootleg Series Vol. 11: The Basement Tapes Complete.
    Bob Dylan and Johnny Cash also recorded a version of the song together at the 1969 Dylan-Cash sessions. This version was officially released on November 1, 2019, on The Bootleg Series Vol. 15: Travelin' Thru, 1967–1969.
    Johnny Cash was featured in a cover performed by The Highwaymen, a country supergroup featuring Cash, Willie Nelson, Waylon Jennings and Kris Kristofferson. This cover is slightly more upbeat, skewing to "Outlaw Country," and features the verse Cash omitted when he first recorded the song for Sun (Jennings sang the verse on the studio recording and in live performances, which allowed each member of the group to sing a verse).
    Tim Armstrong covered the song in 2012 during his Guitar Center session.
    Rosanne Cash recorded a cover of the song on her 1980 album Right or Wrong.
    Beat Farmers recorded a raucous cover of the song on their 1987 album The Pursuit of Happiness.
    Johnny Rivers on the album Memphis Sun Recordings.
    Gene Summers included the song on his "Country Song Roundup" album in 2018.
    Infamous Stringdusters recorded a cover of the song on their 2015 Undercover album.
    Tim Buckley recorded an extended-length improvisational cover in 1968 that appears on the 2-CD set "Live at The Electric Theater Co. Chicago, 1968".
    Bill Monroe performed a version that appears on the compilation album, Down for Double, released in November 2018.
    Lemmy performed a version appearing as the first track on The Head Cat album Lemmy, Slim Jim, & Danny B released in 2000.
    Boyd Holbrook recorded a cover of the song for the 2024 film, A Complete Unknown.

References

Stewart, Brett (December 8, 2023). "Top 10 Johnny Cash Songs". Classic Rock History. Retrieved May 21, 2024. "In 1958, Cash released a single called 'Big River,' a rockabilly song that further concreted his stardom as it shot to the top of the charts for quite some time."
"Wiki Music Guide". Archived from the original on 2013-08-01. Retrieved 2009-08-16.
"Big River". Stevenmenke.com. Retrieved 26 April 2021.
"The SetList Program - Grateful Dead Setlists, Listener Experiences, and Statistics". Setlists.net. Retrieved 26 April 2021.

    "Secret Sisters cover". YouTube. Retrieved 2011-01-28.

    vte

Johnny Cash

    Albums Singles Sun Records Songs Awards

Studio albums	
1950s	

    Johnny Cash with His Hot and Blue Guitar! The Fabulous Johnny Cash Hymns by Johnny Cash Songs of Our Soil

1960s	

    Ride This Train Now, There Was a Song! Hymns from the Heart The Sound of Johnny Cash Blood, Sweat and Tears The Christmas Spirit I Walk the Line Bitter Tears: Ballads of the American Indian Orange Blossom Special Johnny Cash Sings the Ballads of the True West Everybody Loves a Nut Happiness Is You Carryin' On with Johnny Cash & June Carter From Sea to Shining Sea The Holy Land

1970s	

    Hello, I'm Johnny Cash Man in Black A Thing Called Love America: A 200-Year Salute in Story and Song The Johnny Cash Family Christmas Any Old Wind That Blows Johnny Cash and His Woman Ragged Old Flag The Junkie and the Juicehead Minus Me The Johnny Cash Children's Album Johnny Cash Sings Precious Memories John R. Cash Look at Them Beans One Piece at a Time The Last Gunfighter Ballad The Rambler I Would Like to See You Again Gone Girl Silver A Believer Sings the Truth Johnny Cash Sings with B.C. Goodpasture Christian School

1980s	

    Rockabilly Blues Classic Christmas The Baron The Adventures of Johnny Cash Johnny 99 Highwayman Rainbow Class of '55: Memphis Rock & Roll Homecoming Heroes Believe in Him Johnny Cash Is Coming to Town Classic Cash Water from the Wells of Home

1990s	

    Highwayman 2 Boom Chicka Boom The Mystery of Life Country Christmas American Recordings The Road Goes on Forever American II: Unchained

2000–2020s	

    American III: Solitary Man American IV: The Man Comes Around My Mother's Hymn Book American V: A Hundred Highways American VI: Ain't No Grave Out Among the Stars Songwriter

Live albums	

    At Folsom Prison At San Quentin The Johnny Cash Show På Österåker Strawberry Cake The Survivors Koncert v Praze (In Prague – Live) VH1 Storytellers: Johnny Cash & Willie Nelson At Madison Square Garden A Concert Behind Prison Walls Live from Austin, TX

Soundtracks	

    I Walk the Line Little Fauss and Big Halsy The Gospel Road Return to the Promised Land

Compilations	

    Johnny Cash Sings the Songs That Made Him Famous Greatest! Johnny Cash Sings Hank Williams Now Here's Johnny Cash All Aboard the Blue Train with Johnny Cash Original Sun Sound of Johnny Cash Ring of Fire: The Best of Johnny Cash Greatest Hits, Vol. 1 Heart of Cash Greatest Hits, Vol. 2 Sunday Morning Coming Down International Superstar Five Feet High and Rising Destination Victoria Station Greatest Hits, Vol. 3 The Unissued Johnny Cash Johnny & June Tall Man Encore Biggest Hits The Man in Black 1954–1958 The Man in Black 1959–1962 Come Along and Ride This Train The Essential Johnny Cash (1992) Wanted Man The Man in Black 1963–1969 The Man in Black – His Greatest Hits 16 Biggest Hits Love, God, Murder The Essential Johnny Cash (2002) 20th Century Masters – The Millennium Collection: The Best of Johnny Cash Unearthed The Legend The Legend of Johnny Cash Patriot 16 Biggest Hits: Johnny Cash & June Carter Cash The Legend of Johnny Cash Vol. II The Complete Columbia Album Collection Johnny Cash and the Royal Philharmonic Orchestra

Songs	

    "25 Minutes to Go" "Any Old Wind That Blows" "Austin Prison" "A Wonderful Time Up There" "The Ballad of Boot Hill" "The Ballad of Ira Hayes" "Ballad of a Teenage Queen" "Big River" "Blistered" "A Boy Named Sue" "Busted" "Cat's in the Cradle" "Cocaine Blues" "Cry! Cry! Cry!" "Daddy Sang Bass" "Dark as a Dungeon" "Don't Take Your Guns to Town" "Engine 143" "Everybody Loves a Nut" "Flesh and Blood" "The Folk Singer" "Folsom Prison Blues" "Forty Shades of Green" "Get Rhythm" "Give My Love to Rose" "Goodnight, Irene" "Green, Green Grass of Home" "Guess Things Happen That Way" "Hey, Porter" "Home of the Blues" "Hurt" "I Couldn't Keep from Crying" "I Love You Because" "I Still Miss Someone" "I Walk the Line" "I Wish I Was Crazy Again" "If I Had a Hammer" "If I Were a Carpenter" "In My Life" "In the Jailhouse Now" "It Ain't Me Babe" "Jackson" "Kate" "Last Night I Had the Strangest Dream" "The Man Comes Around" "Man in Black" "The Matador" "Oh Lonesome Me" "One Piece at a Time" "Oney" "Orange Blossom Special" "The One on the Right Is on the Left" "Personal Jesus" "Ragged Old Flag" "Remember the Alamo" "(Ghost) Riders in the Sky" "Ring of Fire" "Rock Island Line" "Seasons of My Heart" "So Doggone Lonesome" "Sunday Mornin' Comin' Down" "Tennessee Flat Top Box" "There Ain't No Good Chain Gang" "A Thing Called Love" "Understand Your Man" "What Is Truth" "What'd I Say" "Without Love" "You Are My Sunshine"

Film	
Gospel Road: A Story of Jesus
Television	
The Johnny Cash Show
Biographies	

    Man in Black: His Own Story in His Own Words Cash: The Autobiography Johnny Cash! The Man, His World, His Music My Father and the Man in Black Walk the Line Ring of Fire (musical) Ring of Fire (2013 film) My Darling Vivian

Tribute albums	

    The Sound Behind Johnny Cash Kindred Spirits: A Tribute to the Songs of Johnny Cash Dressed in Black: A Tribute to Johnny Cash Johnny's Blues: A Tribute to Johnny Cash Walk the Line: Original Motion Picture Soundtrack Fade to Black: Memories of Johnny All Aboard: A Tribute to Johnny Cash Johnny Cash Remixed We Walk the Line: A Celebration of the Music of Johnny Cash Forever Words

Associated acts	

    The Highwaymen The Tennessee Three Bob Wootton Carl Perkins W. S. Holland Marshall Grant Luther Perkins Million Dollar Quartet The Great Eighties Eight

Family	

    June Carter Cash John Carter Cash Rosanne Cash Cindy Cash Carlene Carter Tommy Cash Carter Family

Historical sites	

    Johnny Cash Boyhood Home
        Dyess, Arkansas Johnny Cash Museum House of Cash Carter Family Fold

Legacy	
Statue of Johnny Cash
Category
Authority control databases Edit this at Wikidata	

    MusicBrainz work

Categories:

    1958 singlesJohnny Cash songsGrateful Dead songsSongs written by Johnny CashRock-and-roll songsRockabilly songsSong recordings produced by Sam PhillipsSong recordings produced by Jack ClementSun Records singlesThe Highwaymen (country supergroup) songsGene Summers songs1958 songsSongs about the Mississippi River

    This page was last edited on 11 June 2025, at 19:23 (UTC).
    Text is available under the Creative Commons Attribution-ShareAlike 4.0 License; additional terms may apply. By using this site, you agree to the Terms of Use and Privacy Policy. Wikipedia® is a registered trademark of the Wikimedia Foundation, Inc., a non-profit organization.

    Privacy policy
    About Wikipedia
    Disclaimers
    Contact Wikipedia
    Code of Conduct
    Developers
    Statistics
    Cookie statement
    Mobile view

    Wikimedia Foundation
    Powered by MediaWiki

"""
    file = NamedTemporaryFile()
    with open(file.name, 'w') as f:
        f.write(data)
    return file


def test_retrieve(docfile):
    eng = get_sqlalchemy_engine()
    FTS_WEIGHT = 0.75
    try:
        load_document(eng, TestEmbeddingSource(), SemchunkChunker(chunk_size=10, overlap=0.0), docfile)
        i = 0
        last_score = 10.0**99
        last_rank = -1
        for row in retrieve(eng, "Big River", TestEmbeddingSource(), limit=5, fts_weight=FTS_WEIGHT):
            assert isinstance(row, RetrievalResult)
            # All test embeddings are the same, and should equal max, or aren't present, and should equal min
            assert round(row.cosine_similarity_score_normed*10) == 10 or round(row.cosine_similarity_score_normed*10) == 0.0
            for score in (row.bm25_score_normed, row.cosine_similarity_score_normed):
                assert isinstance(score, float)
                assert score >= 0
                assert score <= 1
            
            assert row.rank > last_rank
            last_rank = row.rank

            total_score = FTS_WEIGHT * row.bm25_score_normed + (1-FTS_WEIGHT)*row.cosine_similarity_score_normed
            assert total_score <= last_score
            last_score = total_score
            i += 1

        assert i == 5

        with raises(ProgrammingError):
            with eng.connect() as conn:
                conn.execute(text("SELECT * FROM hnsw_results;"))

    except Exception:
        raise
    finally:
        os.remove(settings.db_path)


def test_vector_search_table_creation(docfile):
    """Verify vector_search creates temp table with correct schema."""
    eng = get_sqlalchemy_engine()
    try:
        load_document(eng, TestEmbeddingSource(), SemchunkChunker(chunk_size=10, overlap=0.0), docfile)

        with eng.begin() as conn:
            # Create vector search temp table
            query_emb = TestEmbeddingSource().get_embedding("test query")
            table_name = vector_search(conn, query_emb, limit=10).__tablename__

            # Verify table exists and has correct schema
            result = conn.execute(text(f"SELECT * FROM {table_name} LIMIT 1")).fetchone()
            assert result is not None
            assert len(result) == 3  # chunk_id, content, score

            # Verify scores are raw (not normalized)
            all_scores = conn.execute(
                text(f"SELECT score FROM {table_name}")
            ).fetchall()
            assert all(score[0] > 0 for score in all_scores)  # Positive scores

        # Verify cleanup after transaction
        with raises(ProgrammingError):
            with eng.connect() as conn:
                conn.execute(text(f"SELECT * FROM {table_name}"))

    finally:
        os.remove(settings.db_path)


def test_fts_search_table_creation(docfile):
    """Verify fts_search creates temp table with correct schema."""
    eng = get_sqlalchemy_engine()
    try:
        load_document(eng, TestEmbeddingSource(), SemchunkChunker(chunk_size=10, overlap=0.0), docfile)

        with eng.begin() as conn:
            # Create FTS search temp table
            table_name = fts_search(conn, "Big River", limit=10).__tablename__

            # Verify table exists and has correct schema
            result = conn.execute(text(f"SELECT * FROM {table_name} LIMIT 1")).fetchone()
            assert result is not None
            assert len(result) == 3  # chunk_id, content, score

            # Verify scores are raw BM25 scores
            all_scores = conn.execute(
                text(f"SELECT score FROM {table_name}")
            ).fetchall()
            assert all(score[0] > 0 for score in all_scores)  # Positive BM25 scores

        # Verify cleanup after transaction
        with raises(ProgrammingError):
            with eng.connect() as conn:
                conn.execute(text(f"SELECT * FROM {table_name}"))

    finally:
        os.remove(settings.db_path)


def test_independent_search_results(docfile):
    """Verify vector and FTS searches can be called independently."""
    eng = get_sqlalchemy_engine()
    try:
        load_document(eng, TestEmbeddingSource(), SemchunkChunker(chunk_size=10, overlap=0.0), docfile)

        with eng.begin() as conn:
            # Call both search functions
            query_emb = TestEmbeddingSource().get_embedding("Big River")
            vector_table = vector_search(conn, query_emb, limit=5).__tablename__
            fts_table = fts_search(conn, "Big River", limit=5).__tablename__

            # Verify both tables exist simultaneously
            vector_results = conn.execute(text(f"SELECT COUNT(*) FROM {vector_table}")).fetchone()
            fts_results = conn.execute(text(f"SELECT COUNT(*) FROM {fts_table}")).fetchone()

            assert vector_results[0] > 0
            assert fts_results[0] > 0

            # Verify they have identical schemas (can be joined)
            vector_cols = conn.execute(text(f"DESCRIBE {vector_table}")).fetchall()
            fts_cols = conn.execute(text(f"DESCRIBE {fts_table}")).fetchall()

            assert len(vector_cols) == len(fts_cols)
            for v_col, f_col in zip(vector_cols, fts_cols):
                assert v_col[0] == f_col[0]  # Column name matches

    finally:
        os.remove(settings.db_path)


def test_retrieve_no_bm25_match(docfile):
    """Verify behavior when BM25 query matches nothing."""
    eng = get_sqlalchemy_engine()
    try:
        load_document(eng, TestEmbeddingSource(), SemchunkChunker(chunk_size=10, overlap=0.0), docfile)

        # Query for something that doesn't exist in BM25 index
        results = retrieve(eng, "xyzabc123impossible", TestEmbeddingSource(), limit=5)

        # Should not crash and should return results from vector search
        assert len(results) > 0  # Vector search returns results
        assert len(results) == 5  # Requested limit

        # BM25 raw scores should be very low (close to default 0.0001) for non-matching queries
        for result in results:
            assert isinstance(result, RetrievalResult)
            assert result.bm25_score < 0.01  # Raw BM25 score is low/default
            assert result.cosine_similarity_score >= -1.0  # Valid cosine similarity range
            assert result.cosine_similarity_score <= 1.0

    finally:
        os.remove(settings.db_path)
