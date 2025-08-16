import argparse
import logging

def record_cmd(args):
    """Record command handler"""
    from .record import record
    
    logging.basicConfig(level=logging.INFO)
    record(args.output, args.duration, trim_silence=not args.no_vad)

def compare_cmd(args):
    """Compare command handler"""  
    from .compare import compare
    
    logging.basicConfig(level=logging.INFO)
    compare(args.file1, args.file2, args.method)

def listen_cmd(args):
    """Listen command handler"""
    from .listen import listen
    
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level)

    listen(args.reference, args.threshold, args.method, 
           args.buffer_size, args.slide_size)

def main():
    parser = argparse.ArgumentParser(prog="lwake")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Record subcommand
    record_parser = subparsers.add_parser("record", help="Record audio samples")
    record_parser.add_argument("output", help="Output .wav file path")
    record_parser.add_argument("--duration", type=int, default=3, help="Duration in seconds")
    record_parser.add_argument("--no-vad", action="store_true", help="Skip VAD silence trimming")
    record_parser.set_defaults(func=record_cmd)

    # Compare subcommand
    compare_parser = subparsers.add_parser("compare", help="Compare two audio files")
    compare_parser.add_argument("file1", help="Path to first audio file")
    compare_parser.add_argument("file2", help="Path to second audio file")
    compare_parser.add_argument("--method", choices=["mfcc", "embedding"], default="embedding", 
                               help="Feature extraction method (default: embedding)")
    compare_parser.set_defaults(func=compare_cmd)

    # Listen subcommand  
    listen_parser = subparsers.add_parser("listen", help="Real-time wake word detection")
    listen_parser.add_argument("reference", help="Folder with reference wake word .wav files")
    listen_parser.add_argument("threshold", type=float, help="DTW cosine distance threshold for detection")
    listen_parser.add_argument("--method", choices=["mfcc", "embedding"], default="embedding", 
                              help="Feature extraction method (default: embedding)")
    listen_parser.add_argument("--buffer-size", type=float, default=2.0, 
                              help="Audio buffer size in seconds (default: 2.0)")
    listen_parser.add_argument("--slide-size", type=float, default=0.25, 
                              help="Slide size in seconds (default: 0.25)")
    listen_parser.add_argument("--debug", action="store_true", 
                              help="Print debug messages to stderr")
    listen_parser.set_defaults(func=listen_cmd)

    args = parser.parse_args()
    args.func(args)