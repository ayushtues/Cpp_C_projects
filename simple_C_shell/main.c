#include <sys/wait.h> 
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>


// forward declarations for the built in shell commands
int lsh_cd(char **args);
int lsh_help(char **args);
int lsh_exit(char **args);

/*
    List of bulitin commands, followed by their corresponding functions
*/
char *builtin_str[] = {
    "cd",
    "help",
    "exit",
};

int (*builtin_func[])(char **) = {
    &lsh_cd,
    &lsh_help,
    &lsh_exit
};

int lsh_num_builtins(){
    return sizeof(builtin_str) / sizeof(char*);
}

int lsh_cd(char **args)
{
    if (args[1] == NULL)
    {
        fprintf(stderr, "lsh: expected argument to \"cd\"\n");
    }
    else
    {
        if(chdir(args[1]) != 0 )
        {
            perror("lsh");
        }
    }
    return 1;
    
}

int lsh_help(char **args)
{
    int i;
    printf("Ayush Mangal's LSH\n");
    printf("Type program names and arguments, and hit enter.\n");
    printf("The following are built in: \n");
    
    for (i = 0; i< lsh_num_builtins(); i++)
    {
        printf(" %s\n", builtin_str[i]);
    }

    printf("Use the man command for information on other programs. \n");
    return 1 ;
}

int lsh_exit(char **args)
{
    return 0; // return 0 to signal exit
}

#define LSH_RL_BUFSIZE 1024
char *lsh_read_line(void)
{
    int bufsize = LSH_RL_BUFSIZE;
    int position = 0;

    // allocate the buffer
    char *buffer = malloc(sizeof(char) * bufsize);
    int c;

    if(!buffer)
    {
        fprintf(stderr, "lsh: allocation error\n");
        exit(EXIT_FAILURE);
    }

    while(1)
    {
        // Read a character
        c = getchar();

        // If we hit EOF, replace it with a null character and return
        if (c == EOF || c == '\n')
        {
            buffer[position] = '\0';
            return buffer;
        }
        else
        {   // if not EOF, simply store the character read
            buffer[position] = c;
        }

        // increment position
        position++;

        if( position >= bufsize)
        {   
            // dynamically increase the size of the buffer if max size reached
            bufsize += LSH_RL_BUFSIZE;
            buffer = realloc(buffer, bufsize);
            if(!buffer)
            {
                fprintf(stderr, "lsh: allocation error \n");
                exit(EXIT_FAILURE);
            }
        }
    }
}

#define LSH_TOK_BUFSIZE 64
#define LSH_TOK_DELIM " \t\r\n\a"
char **lsh_split_line(char *line)
{
    int bufsize = LSH_TOK_BUFSIZE;
    int position = 0;
    char **tokens = malloc(bufsize*sizeof(char*));
    char *token;

    if(!tokens)
    {   
        fprintf(stderr, "lsh: allocation error\n");
        exit(EXIT_FAILURE);
    }

    // returns pointers to within the string, after tokenising it and adding \0 bytes at the end of each token
    token = strtok(line, LSH_TOK_DELIM);
    while( token != NULL)
    {
        tokens[position] = token;
        position++;

        if(position >= bufsize)
        {
            bufsize += LSH_TOK_BUFSIZE;
            tokens = realloc(tokens, bufsize * sizeof(char *));

            if(!tokens)
            {   
                fprintf(stderr, "lsh: allocation error\n");
                exit(EXIT_FAILURE);
            }
            
        }

        token = strtok(NULL, LSH_TOK_DELIM);
    }

    tokens[position] = NULL;
    return tokens;    

}

int lsh_launch(char **args)
{
    pid_t pid, wpid;
    int status;

    pid = fork();
    if(pid ==0) // child process
    {
        if(execvp(args[0], args) == -1){    // if the child process returns, there's a failure
            perror("lsh");
        }
        exit(EXIT_FAILURE);
    }
    else if (pid < 0 ) // error in forking
    {
        perror("lsh");
    }
    else
    {
        do{
            wpid = waitpid(pid, &status, WUNTRACED);
        }while(!WIFEXITED(status) && !WIFSIGNALED(status)); // wait for the child process to either exit or a signal
    }
    return 1;
    
}

int lsh_execute( char** args)
{
    int i;

    if(args[0] == NULL)
    {   // empty command
        return 1;
    }

    for(i = 0; i<lsh_num_builtins(); i++)
    {
        if(strcmp(args[0], builtin_str[i]) == 0)
        {
            return (*builtin_func[i])(args); // if a bultin function matchs with the arg, call it
        }
    }

    return lsh_launch(args); //else call lsh_launch;
}

void lsh_loop(void)
{
    char *line;
    char **args;
    int status;

    do{
        printf("> ");
        line = lsh_read_line(); // read
        args = lsh_split_line(line); // parse
        status = lsh_execute(args); // execute

        // free resources
        free(line);
        free(args);
    }while(status);
}

int main(int argc, char **argv)
{
    // load config files, if any
    printf("Welcome to Ayush's LSH \n");
    // run command loop
    lsh_loop();

    printf("Have a good day!");

    // perform any shutdown/cleanup
    return EXIT_SUCCESS;


}