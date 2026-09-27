"""Compatibility module; CSI owns the installed ecat-psgrn command."""


def main():
    from csi.cli_tools.psgrn_cli import main as csi_main
    return csi_main()


if __name__ == "__main__":
    main()
