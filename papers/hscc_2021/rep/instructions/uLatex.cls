%-------------------------------------------------------------------------------
% Class uLatex is based on ltxdoc
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{uLatex}[2019/11/22 v1.0 uLatex documentation]
\LoadClass[10pt,a4paper]{ltxdoc}
%-------------------------------------------------------------------------------
% Encoding: utf8 and T1
\RequirePackage[utf8]{inputenc}
\RequirePackage[T1]{fontenc}
%-------------------------------------------------------------------------------
% Standards packages
\RequirePackage{varioref}
\RequirePackage{varwidth}
\RequirePackage{keystroke}
\RequirePackage{verbatim}
\RequirePackage{booktabs}
\RequirePackage{tabularx}
\RequirePackage{xspace}
\RequirePackage{hologo}
\RequirePackage{mwe}
%-------------------------------------------------------------------------------
% Get and print information on the git repository
%\RequirePackage[maxdepth=5,mark]{gitinfo2}
%\RequirePackage{Ugitinfo}
%-------------------------------------------------------------------------------
% Font and color package
\RequirePackage[setup]{Ucolor}
\RequirePackage[text=ubermv]{Ufont}
\UsetColor{caption label=black}
%-------------------------------------------------------------------------------
% Package tcolorbox is copiously used
\RequirePackage[all]{tcolorbox}
% settings for docCommand, docKey, docColor, dispExample, dispListing
\tcbset{%
  doc left indent=-1.5cm,
  doc head={colback=UPlot25,interior style=fill},
  color option=UGreen,
}
\tcbset{%
  docexample/.style={%
    boxrule=1pt,
    colframe=UBlue,
    colback=white,
    arc=0pt,
    fonttitle=\bfseries,
    listing engine=minted,
  },
  usage/.style={%
    detach title,
    before upper={\tcbtitle\quad},
    coltitle=UBlue,
    title={Usage},
  },
  documentation minted options={%
    fontsize=\normalsize,
  },
}
\tcbset{%
  color definition=UOrange,
  color key=UGreen,
  color value=UBlue,
  color command=UBlue,
  color color=UOrange,
  color hyperlink=UBlue,
}
%\tcbset{}
% settings for the index \tcbset{index format=doc}
\tcbset{index colorize}
\usemintedstyle{trac}
\newcommand{\UkeyPath}[1]{\textcolor{UGreen}{\ttfamily#1}}
%-------------------------------------------------------------------------------
% Page layout
\RequirePackage[left=3cm,right=3cm,top=2cm,bottom=3cm]{geometry}
\RequirePackage{marginnote}
\setlength\parindent{0pt}
%-------------------------------------------------------------------------------
% Case Manipulation
\RequirePackage{mfirstuc}
\newcommand\firstToLow[1]{%
  {%
    \renewcommand{\mfirstucMakeUppercase}{\MakeLowercase}%
    \makefirstuc{#1}%
  }%
}
%-------------------------------------------------------------------------------
% Bibliography
\RequirePackage[style=nature]{Ubiblio}
\IfFileExists{main.bib}{%
    \addbibresource{main.bib}
  }{%
    \ClassWarning{uLatex}{File main.bib not found}
}
%-------------------------------------------------------------------------------
% (Cross-)references
\RequirePackage{Uref}
\hypersetup{colorlinks=true,runcolor=UGreen}
\UwritePdfMetaProperties{Created with uLatex}
%-------------------------------------------------------------------------------
% Write loaded packages information in log file
\listfiles
%-------------------------------------------------------------------------------
% Record index informations
\makeindex
\DisableCrossrefs
\RecordChanges
%-------------------------------------------------------------------------------
% Font and colors settings
\UsetColor{section=black,subsection=black,subsubsection=black}
\renewcommand{\ttdefault}{lmtt}
\def\theCodelineNo{\textcolor{UBlue}{\sffamily\footnotesize\oldstylenums{\arabic{CodelineNo}}}}
%-------------------------------------------------------------------------------
% Shortcuts
\newcommand{\miktex}{\hologo{MiKTeX}\xspace}
\newcommand{\texlive}{\hologo{TeX}Live\xspace}
\newcommand\TeXstudio{\hologo{TeX}studio\xspace}
\newcommand{\mactex}{Mac\hologo{TeX}}
%-------------------------------------------------------------------------------
% Maketitle
\def\@subtitle{}
\providecommand{\subtitle}[1]{\gdef\@subtitle{#1}}
\let\OLDmaketitle\maketitle
\renewcommand{\maketitle}[1][package]{%
  \pdfbookmark{\@title}{Utitledocanchor}
  \begin{tcolorbox}[
    size=tight,oversize,
    sharp corners,
    enhanced,
    colframe=UBlack,
    colback=UBlack,
    colupper=white,
    enlarge left by=-100pt,
    right=160pt,
    left=30pt,
    bottom=40pt,
    top=10pt,
    fontupper={\Huge\bfseries},
    overlay={%
      \node[white,xshift=5pt] at (frame.center) {\bfseries\Large \@title};
    },
  ]
  \end{tcolorbox}
  \begin{center}
  \ifthenelse{\equal{#1}{package}}{%
  \begin{minipage}{0.8\textwidth}
    \begin{tabular}{rl}
      {Package Version} & \UinsertGitVersion \\[0.5ex]
      {Project home site}    & \url{http://t.uber.com/ulatex} \\[0.5ex]
      {Release date}    &  \UinsertGitDate \\[0.5ex]
      {Git reference}   & \UinsertGitBranch\,@\,\gitFirstTagDescribe \\[0.5ex]
      {Contact}         & \docEmail{\gitAuthorEmail}
    \end{tabular}
  \end{minipage}
  }{%
      {\bfseries\Large\@subtitle}\\[1ex]
      \@author\\[1ex]
      \@date
  }
  \end{center}
  \vskip-1cm
}%
%-------------------------------------------------------------------------------
\renewcommand{\contentsname}{}
\newenvironment{doctable}[1]{%
  \begin{tabular}{@{}#1@{}}
    \toprule
  }{%
    \bottomrule
  \end{tabular}
}
%-------------------------------------------------------------------------------
\newtcbox{todo}{enhanced
  ,capture=minipage
  ,frame hidden
  ,oversize
  ,interior hidden
  ,before upper={\color{URed}\sffamily \marginnote{To Do}}
}
%-------------------------------------------------------------------------------
\newtcolorbox{boxtitlepage}[1]{
  title={#1},
  colback=UPlot64,
  colframe=UPlot62,
  fonttitle=\bfseries,
  size=small
}
%-------------------------------------------------------------------------------
\newcommand{\docEmail}[1]{\href{#1}{#1}\xspace}
%-------------------------------------------------------------------------------
\newcommand{\InsertLicenseBlaBla}{%
  Copyright 2019 Uber Technologies, Inc.
  All rights reserved.
}
%-------------------------------------------------------------------------------
\newcommand{\InsertPgfkeysBlaBla}[1]{%
  \def\UdocuAtUsetName{Uset#1}
  \def\Udocu@package{\firstToLow{#1}}
  \begin{hint}
    Package \UpackageBox{U\Udocu@package} uses \PackageName{pgfkeys} internally to create option and parameter keys. In the
    following documentation all the defined keys are presented with their whole path, but only their names is needed
    in order for you to use them: the path \UkeyPath{/U/\Udocu@package/} must be ignored when used in
    \expandafter\refCom\expandafter{\UdocuAtUsetName}
  \end{hint}
}
%-------------------------------------------------------------------------------
\newtcolorbox{warning}{skin=enhancedmiddle jigsaw,breakable,parbox=false,
  boxrule=0pt,leftrule=3mm,rightrule=3mm,boxsep=0pt,arc=0pt,outer arc=0pt,
  left=3mm,right=3mm,top=1mm,bottom=1mm,toptitle=1mm,bottomtitle=1mm,oversize,
  colback=UPlot25,colframe=UOrange
}
%-------------------------------------------------------------------------------
\newtcolorbox{hint}{skin=enhancedmiddle jigsaw,breakable,parbox=false,
  boxrule=0pt,leftrule=3mm,rightrule=3mm,boxsep=0pt,arc=0pt,outer arc=0pt,
  left=3mm,right=3mm,top=1mm,bottom=1mm,toptitle=1mm,bottomtitle=1mm,oversize,
  colback=UBlue!5!white,colframe=UBlue}
%-------------------------------------------------------------------------------
\newcommand{\InsertImplementationBlabla}{%
 \begin{warning}
 The following section does not aim to explain the usage of this package.
 It is intended for the package writer or anyone who want to understand the
 code and participate to its development.
 \end{warning}
}
%-------------------------------------------------------------------------------
\newcommand{\PrintBibliography}{%
  \phantomsection
  \cleardoublepage
  \addcontentsline{toc}{section}{References}
  \printbibliography
}
%-------------------------------------------------------------------------------
\AtEndDocument{%
    \phantomsection
    \cleardoublepage
    \PrintIndex
    \phantomsection
    \cleardoublepage
    \addcontentsline{toc}{section}{Change History}
    \PrintChanges
}
%-------------------------------------------------------------------------------
% From basic-code.sty. ENV bashshell
\newtcblisting{bashshell}[1][]{
  colback=UGray!5!white,
  colupper=black,
  colframe=black,
  top=0pt,bottom=0pt,
  listing engine=listings,
  listing only,boxrule=0.6pt,
  listing options={
    style=tcblatex,
    language=sh,
    keywordstyle={\color{black}\bfseries},
    escapeinside={(*@}{@*)},
    morekeywords={ls,ln,rm,mkdir,git,cp,grep,ldd,qmake,make,source,sudo,chmod,wget}
  },
  every listing line*={\textcolor{UGray}{\ttfamily\bfseries \$> }},
  fonttitle=\bfseries,
  size=fbox,
  #1
}
%-------------------------------------------------------------------------------
\newcommand{\InsertDocCompilationBlaBla}{%
  The present documentation was compiled
  Required to compile the documentation: (not all of this is necessary needed to use Ulatex)
  tcolorbox (version 3.6 at last!!) (for library documentation features)
  pgf (recent and well installed)
}
%-------------------------------------------------------------------------------
\newtcbox{\FileName}{enhanced,nobeforeafter,tcbox raise base,boxrule=0.4pt,top=0mm,bottom=0mm,frame hidden,
  right=0mm,left=0mm,arc=0pt,outer arc=0pt,boxrule=0pt,boxsep=1pt,before upper={\vphantom{dlg}}\ttfamily,
  ,coltext=UGreen,colback=white
}
%-------------------------------------------------------------------------------
\DeclareTotalTCBox{\cmdbox}{ s O{>} v }{
    before upper={\vphantom{dlg}},
    verbatim,boxsep=0pt,tcbox raise base,
    colframe=UGray,colupper=black,
    colback=UGray!5!white,
    top=2.5pt,bottom=1pt,boxrule=0.4pt,left=2pt,right=2pt,arc=2pt,
  }{%
    \IfBooleanTF{#1}{}{\textcolor{UGray}{\ttfamily\bfseries #2\,}}%
    \lstinline[%
    language=bash,morekeywords={tex,latex,pdflatex},
     keywordstyle={\color{UGray}\bfseries}
    ]^#3^
}
%-------------------------------------------------------------------------------
\newtcbox{\flagbox}{
  on line
  ,before upper={\vphantom{dlg}\sffamily}
  ,boxrule=0.4pt,top=0pt,bottom=0mm,right=2pt,left=2pt,arc=1pt,boxsep=1pt,
  colframe=UBlue,coltext=UBlue,colback=UBlue!10!white
}
%-------------------------------------------------------------------------------
\newtcbox{\cmdarg}[1][UOrange]{enhanced,nobeforeafter,tcbox raise base,boxrule=0.4pt,top=0mm,bottom=0mm,
  right=0mm,left=0mm,arc=0pt,outer arc=0pt,boxrule=0pt,boxsep=1pt,before upper={\vphantom{dlg}}\ttfamily,
  ,coltext=#1,colback=white,frame hidden
}
%-------------------------------------------------------------------------------
\newcommand{\UlinkDocumentation}[1]{\href[pdfnewwindow]{run:./#1.pdf}{#1}}
\newtcbox{\UpackageBox}{enhanced,
  nobeforeafter,tcbox raise base,boxrule=0.4pt,top=0.5mm,bottom=0mm,
  right=0mm,left=4mm,arc=1pt,boxsep=1pt,before upper={\vphantom{dlg}},
  colframe=UGreen,coltext=UGreen,colback=white,fontupper=\normalfont,
  overlay={%
    \begin{tcbclipinterior}
      \fill[UGreen] (frame.south west) rectangle
        node[text=white,scale=0.85,font=\sffamily\bfseries\tiny,rotate=90] {PKG} ([xshift=4mm]frame.north west);
    \end{tcbclipinterior}
  }
}
\newcommand{\Upackage}[1]{\UpackageBox{\UlinkDocumentation{#1}}}
%-------------------------------------------------------------------------------
\newcommand{\extension}[1]{\texttt{#1}}
\newcommand{\PackageName}[1]{\textsf{\textcolor{UPlot64}{#1}}}
\newcommand{\DirPath}[1]{\texttt{\textcolor{UGreen}{#1}}}
\newcommand{\EnvVariable}[1]{\texttt{\textcolor{UBlue}{\$#1}}}
\newcommand{\UdocClr}[1]{ \draw[fill=#1,draw=white] (0,0) rectangle (1.618,1); }
\newcommand{\UdocLbl}[1]{ \node[right,font=\ttfamily] at (0,0.5) {#1}; }
%-------------------------------------------------------------------------------
