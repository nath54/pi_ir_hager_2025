#
from typing import Optional
#
from lib_test import Tester

#
TESTS: list[ tuple[str, tuple[int, ...], str, Optional[str]] ] = [

    ("simple_linear_model.py", (2, 10), "SimpleLinearModel", None),
    ("simple_linear_model_2.py", (2, 10), "SimpleLinearModel", None),
    ("complex_model.py", (1, 3, 224, 224), "ComplexModel", None),
    ("simple_conv_model.py", (2, 3, 32, 32), "SimpleConvModel", None),

    ("test_model_architecture_1_2.py", (4, 10), "Model", None),
    ("test_model_architecture_1.py", (4, 10), "Model", None),
    ("test_model_architecture_2_1.py", (4, 20), "Model", None),
    ("test_model_architecture_3_1.py", (4, 20), "Model", None),

    # ...

]


#
MSGL: int = 60
FILEL: int = 40


#
if __name__ == "__main__":

    # Couleurs ANSI
    class Colors:
        RESET = '\033[0m'
        BOLD = '\033[1m'
        RED = '\033[91m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        BLUE = '\033[94m'
        MAGENTA = '\033[95m'
        CYAN = '\033[96m'
        WHITE = '\033[97m'
        BG_RED = '\033[101m'
        BG_GREEN = '\033[102m'
        BG_YELLOW = '\033[103m'

    def print_header():
        """Affiche l'en-tête du tableau"""
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*120}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'RÉSULTATS DES TESTS - MODELBLOCKS IMPLEMENTATION':^120}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*120}{Colors.RESET}")

    def print_table_header():
        """Affiche l'en-tête du tableau"""
        print(f"\n{Colors.BOLD}{Colors.WHITE}{'│':<1} {'MODÈLE':<{FILEL}} {'DIMENSIONS':<12} {'EXT':<6} {'LNK':<6} {'EXE':<6} {'STATUT':<12} {'MESSAGE':<{MSGL}} {'│'}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.WHITE}{'├'}{'─'*FILEL}{'┬'}{'─'*12}{'┬'}{'─'*6}{'┬'}{'─'*6}{'┬'}{'─'*6}{'┬'}{'─'*12}{'┬'}{'─'*MSGL}{'┤'}{Colors.RESET}")

    def get_status_color(extraction: bool, pytorch: bool, execution: bool) -> tuple[str, str]:
        """Retourne la couleur et le texte du statut"""
        if extraction and pytorch and execution:
            return Colors.GREEN, "✓ SUCCÈS"
        elif extraction and pytorch:
            return Colors.YELLOW, "⚠ PARTIEL"
        elif extraction:
            return Colors.RED, "✗ ÉCHEC"
        else:
            return Colors.RED, "✗ CRITIQUE"

    def wrap_message(message: str, max_len: int = MSGL-2) -> list[str]:
        """Divise le message en plusieurs lignes si nécessaire"""
        if len(message) <= max_len:
            return [message]

        words = message.split()
        lines = []
        current_line = ""

        for word in words:
            if len(current_line + " " + word) <= max_len:
                if current_line:
                    current_line += " " + word
                else:
                    current_line = word
            else:
                if current_line:
                    lines.append(current_line)
                    current_line = word
                else:
                    # Si un seul mot est trop long, on le coupe
                    lines.append(word[:max_len-3] + "...")
                    current_line = ""

        if current_line:
            lines.append(current_line)

        return lines

    def print_test_result(code_path: str, input_dim: tuple, extraction: bool, pytorch: bool, execution: bool, message: str):
        """Affiche le résultat d'un test"""
        # Couleurs pour chaque étape
        ext_color = Colors.GREEN if extraction else Colors.RED
        pyt_color = Colors.GREEN if pytorch else Colors.RED
        exec_color = Colors.GREEN if execution else Colors.RED

        # Statut global
        status_color, status_text = get_status_color(extraction, pytorch, execution)

        # Formatage des dimensions
        dim_str = "x".join(map(str, input_dim))
        if len(dim_str) > 12:
            dim_str = dim_str[:9] + "..."

        # Wrapping du message
        message_lines = wrap_message(message)

        # Affichage de la première ligne
        first_line_msg = message_lines[0] if message_lines else ""
        print(f"{Colors.WHITE}{'│':<1} {code_path:<{FILEL}} {dim_str:<12} {ext_color}{'✓' if extraction else '✗':<6}{Colors.RESET} {pyt_color}{'✓' if pytorch else '✗':<6}{Colors.RESET} {exec_color}{'✓' if execution else '✗':<6}{Colors.RESET} {status_color}{status_text:<12}{Colors.RESET} {first_line_msg:<{MSGL-2}} {'│'}{Colors.RESET}")

        # Affichage des lignes supplémentaires du message
        for i in range(1, len(message_lines)):
            print(f"{Colors.WHITE}{'│':<1} {'':<{FILEL}} {'':<12} {'':<6} {'':<6} {'':<6} {'':<12} {message_lines[i]:<{MSGL-2}} {'│'}{Colors.RESET}")

        # Ligne de séparation entre les tests
        print(f"{Colors.WHITE}{'├'}{'─'*FILEL}{'┼'}{'─'*12}{'┼'}{'─'*6}{'┼'}{'─'*6}{'┼'}{'─'*6}{'┼'}{'─'*12}{'┼'}{'─'*MSGL}{'┤'}{Colors.RESET}")

    def print_footer(total_tests: int, successful: int, partial: int, failed: int):
        """Affiche le pied du tableau avec les statistiques"""
        print(f"{Colors.BOLD}{Colors.WHITE}{'└'}{'─'*FILEL}{'┴'}{'─'*12}{'┴'}{'─'*6}{'┴'}{'─'*6}{'┴'}{'─'*6}{'┴'}{'─'*12}{'┴'}{'─'*MSGL}{'┘'}{Colors.RESET}")

        print(f"\n{Colors.BOLD}{Colors.CYAN}{'STATISTIQUES':^120}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.WHITE}{'─'*120}{Colors.RESET}")

        success_rate = (successful / total_tests * 100) if total_tests > 0 else 0
        partial_rate = (partial / total_tests * 100) if total_tests > 0 else 0
        fail_rate = (failed / total_tests * 100) if total_tests > 0 else 0

        print(f"{Colors.GREEN}✓ Tests réussis:     {successful:>3}/{total_tests} ({success_rate:>5.1f}%){Colors.RESET}")
        print(f"{Colors.YELLOW}⚠ Tests partiels:    {partial:>3}/{total_tests} ({partial_rate:>5.1f}%){Colors.RESET}")
        print(f"{Colors.RED}✗ Tests échoués:     {failed:>3}/{total_tests} ({fail_rate:>5.1f}%){Colors.RESET}")

        print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*120}{Colors.RESET}\n")

    # Compteurs pour les statistiques
    total_tests = len(TESTS)
    successful = 0
    partial = 0
    failed = 0

    # Affichage de l'en-tête
    print_header()
    print_table_header()

    #
    for t in TESTS:

        #
        code_path: str = t[0]
        input_dim: tuple[int, ...] = t[1]
        main_block_name: str = t[2]
        weights_path: Optional[str] = t[3]

        #
        tester: Tester = Tester(
            code_path=f"tests/test_models/{code_path}",
            input_dim=input_dim,
            main_block_name=main_block_name,
            weights_path=weights_path
        )

        #
        res: tuple[bool, bool, bool, str] = tester.test()

        # Extraction des résultats
        extraction_success, pytorch_success, execution_success, message = res

        # Mise à jour des compteurs
        if extraction_success and pytorch_success and execution_success:
            successful += 1
        elif extraction_success and pytorch_success:
            partial += 1
        else:
            failed += 1

        # Affichage du résultat
        print_test_result(code_path, input_dim, extraction_success, pytorch_success, execution_success, message)

    # Affichage du pied avec les statistiques
    print_footer(total_tests, successful, partial, failed)
