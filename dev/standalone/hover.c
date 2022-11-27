/*
 * Copyright (c) 1983-2013 Trevor Wishart and Composers Desktop Project Ltd
 * http://www.trevorwishart.co.uk
 * http://www.composersdesktop.com
 *
 This file is part of the CDP System.

    The CDP System is free software; you can redistribute it
    and/or modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    The CDP System is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with the CDP System; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
    02111-1307 USA
 *
 */



#include <stdio.h>
#include <stdlib.h>
#include <structures.h>
#include <tkglobals.h>
#include <pnames.h>
#include <filetype.h>
#include <processno.h>
#include <modeno.h>
#include <logic.h>
#include <globcon.h>
#include <cdpmain.h>
#include <math.h>
#include <mixxcon.h>
#include <osbind.h>
#include <standalone.h>
#include <ctype.h>
#include <sfsys.h>
#include <string.h>
#include <srates.h>

#if defined unix || defined __GNUC__
#define round(x) lround((x))
#endif
#ifndef HUGE
#define HUGE 3.40282347e+38F
#endif

char errstr[2400];

int anal_infiles = 1;
int	sloom = 0;
int sloombatch = 0;

const char* cdp_version = "7.1.0";

//CDP LIB REPLACEMENTS
static int check_hover_param_validity_and_consistency(dataptr dz);
static int setup_hover_application(dataptr dz);
static int parse_sloom_data(int argc,char *argv[],char ***cmdline,int *cmdlinecnt,dataptr dz);
static int parse_infile_and_check_type(char **cmdline,dataptr dz);
static int setup_hover_param_ranges_and_defaults(dataptr dz);
static int handle_the_outfile(int *cmdlinecnt,char ***cmdline,dataptr dz);
static int setup_and_init_input_param_activity(dataptr dz,int tipc);
static int setup_input_param_defaultval_stores(int tipc,aplptr ap);
static int establish_application(dataptr dz);
static int initialise_vflags(dataptr dz);
static int setup_parameter_storage_and_constants(int storage_cnt,dataptr dz);
static int initialise_is_int_and_no_brk_constants(int storage_cnt,dataptr dz);
static int mark_parameter_types(dataptr dz,aplptr ap);
static int assign_file_data_storage(int infilecnt,dataptr dz);
static int get_tk_cmdline_word(int *cmdlinecnt,char ***cmdline,char *q);
static int get_the_process_no(char *prog_identifier_from_cmdline,dataptr dz);
static int setup_and_init_input_brktable_constants(dataptr dz,int brkcnt);
static int hover(dataptr dz);
static int copy_to_output(int *obufpos,int sampcnt,dataptr dz);

#define HOVER_FRQ	0
#define HOVER_LOC	1
#define HOVER_FRQR	2
#define HOVER_LOCR	3
#define HOVER_SPLIC	4
#define HOVER_DUR	5

#define TRAVERSE	0

/**************************************** MAIN *********************************************/

int main(int argc,char *argv[])
{
	int exit_status;
	dataptr dz = NULL;
	char **cmdline;
	int  cmdlinecnt;
	int n;
	aplptr ap;
	int is_launched = FALSE;
	if(argc==2 && (strcmp(argv[1],"--version") == 0)) {
		fprintf(stdout,"%s\n",cdp_version);
		fflush(stdout);
		return 0;
	}
						/* CHECK FOR SOUNDLOOM */
	if((sloom = sound_loom_in_use(&argc,&argv)) > 1) {
		sloom = 0;
		sloombatch = 1;
	}
	if(sflinit("cdp")){
		sfperror("cdp: initialisation\n");
		return(FAILED);
	}
						  /* SET UP THE PRINCIPLE DATASTRUCTURE */
	if((exit_status = establish_datastructure(&dz))<0) {					// CDP LIB
		print_messages_and_close_sndfiles(exit_status,is_launched,dz);
		return(FAILED);
	}
	if(!sloom) {
		if(argc == 1) {
			usage1();	
			return(FAILED);
		} else if(argc == 2) {
			usage2(argv[1]);	
			return(FAILED);
		}
	}
	if(!sloom) {
		if((exit_status = make_initial_cmdline_check(&argc,&argv))<0) {		// CDP LIB
			print_messages_and_close_sndfiles(exit_status,is_launched,dz);
			return(FAILED);
		}
		cmdline    = argv;
		cmdlinecnt = argc;
		if((get_the_process_no(argv[0],dz))<0)
			return(FAILED);
		cmdline++;
		cmdlinecnt--;
		// setup_particular_application =
		if((exit_status = setup_hover_application(dz))<0) {
			print_messages_and_close_sndfiles(exit_status,is_launched,dz);
			return(FAILED);
		}
		if((exit_status = count_and_allocate_for_infiles(cmdlinecnt,cmdline,dz))<0) {		// CDP LIB
			print_messages_and_close_sndfiles(exit_status,is_launched,dz);
			return(FAILED);
		}
	} else {
		//parse_TK_data() =
		if((exit_status = parse_sloom_data(argc,argv,&cmdline,&cmdlinecnt,dz))<0) {
			exit_status = print_messages_and_close_sndfiles(exit_status,is_launched,dz);
			return(exit_status);		 
		}
	}
	ap = dz->application;

	// parse_infile_and_hone_type() = 
	if((exit_status = parse_infile_and_check_type(cmdline,dz))<0) {
		exit_status = print_messages_and_close_sndfiles(exit_status,is_launched,dz);
		return(FAILED);
	}
	// setup_param_ranges_and_defaults() =
	if((exit_status = setup_hover_param_ranges_and_defaults(dz))<0) {
		exit_status = print_messages_and_close_sndfiles(exit_status,is_launched,dz);
		return(FAILED);
	}
	// open_first_infile		CDP LIB
	if((exit_status = open_first_infile(cmdline[0],dz))<0) {	
		print_messages_and_close_sndfiles(exit_status,is_launched,dz);	
		return(FAILED);
	}
	cmdlinecnt--;
	cmdline++;

//	handle_extra_infiles() : redundant
	// handle_outfile() = 
	if((exit_status = handle_the_outfile(&cmdlinecnt,&cmdline,dz))<0) {
		print_messages_and_close_sndfiles(exit_status,is_launched,dz);
		return(FAILED);
	}

//	handle_formants()			redundant
//	handle_formant_quiksearch()	redundant
//	handle_special_data()		redundant
 
	if((exit_status = read_parameters_and_flags(&cmdline,&cmdlinecnt,dz))<0) {		// CDP LIB
		print_messages_and_close_sndfiles(exit_status,is_launched,dz);
		return(FAILED);
	}
//	check_param_validity_and_consistency....
	if((exit_status = check_hover_param_validity_and_consistency(dz))<0) {
		print_messages_and_close_sndfiles(exit_status,is_launched,dz);
		return(FAILED);
	}
	is_launched = TRUE;
	dz->bufcnt = 4;
	if((dz->sampbuf = (float **)malloc(sizeof(float *) * (dz->bufcnt+1)))==NULL) {
		sprintf(errstr,"INSUFFICIENT MEMORY establishing sample buffers.\n");
		return(MEMORY_ERROR);
	}
	if((dz->sbufptr = (float **)malloc(sizeof(float *) * dz->bufcnt))==NULL) {
		sprintf(errstr,"INSUFFICIENT MEMORY establishing sample buffer pointers.\n");
		return(MEMORY_ERROR);
	}
	for(n = 0;n <dz->bufcnt; n++)
		dz->sampbuf[n] = dz->sbufptr[n] = (float *)0;
	dz->sampbuf[n] = (float *)0;

	if((exit_status = create_sndbufs(dz))<0) {							// CDP LIB
		print_messages_and_close_sndfiles(exit_status,is_launched,dz);
		return(FAILED);
	}
	if(dz->buflen <= dz->iparam[HOVER_SPLIC] * 2) {
		sprintf(errstr,"Buffers too small for splice length\n");
		print_messages_and_close_sndfiles(exit_status,is_launched,dz);
		return(FAILED);
	}
	//param_preprocess()						redundant
	//spec_process_file =
	if((exit_status = hover(dz))<0) {
		print_messages_and_close_sndfiles(exit_status,is_launched,dz);
		return(FAILED);
	}
	if((exit_status = complete_output(dz))<0) {										// CDP LIB
		print_messages_and_close_sndfiles(exit_status,is_launched,dz);
		return(FAILED);
	}
	exit_status = print_messages_and_close_sndfiles(FINISHED,is_launched,dz);		// CDP LIB
	free(dz);
	return(SUCCEEDED);
}

/**********************************************
		REPLACED CDP LIB FUNCTIONS
**********************************************/


/****************************** SET_PARAM_DATA *********************************/

int set_param_data(aplptr ap, int special_data,int maxparamcnt,int paramcnt,char *paramlist)
{
	ap->special_data   = (char)special_data;	   
	ap->param_cnt      = (char)paramcnt;
	ap->max_param_cnt  = (char)maxparamcnt;
	if(ap->max_param_cnt>0) {
		if((ap->param_list = (char *)malloc((size_t)(ap->max_param_cnt+1)))==NULL) {	
			sprintf(errstr,"INSUFFICIENT MEMORY: for param_list\n");
			return(MEMORY_ERROR);
		}
		strcpy(ap->param_list,paramlist); 
	}
	return(FINISHED);
}

/****************************** SET_VFLGS *********************************/

int set_vflgs
(aplptr ap,char *optflags,int optcnt,char *optlist,char *varflags,int vflagcnt, int vparamcnt,char *varlist)
{
	ap->option_cnt 	 = (char) optcnt;			/*RWD added cast */
	if(optcnt) {
		if((ap->option_list = (char *)malloc((size_t)(optcnt+1)))==NULL) {
			sprintf(errstr,"INSUFFICIENT MEMORY: for option_list\n");
			return(MEMORY_ERROR);
		}
		strcpy(ap->option_list,optlist);
		if((ap->option_flags = (char *)malloc((size_t)(optcnt+1)))==NULL) {
			sprintf(errstr,"INSUFFICIENT MEMORY: for option_flags\n");
			return(MEMORY_ERROR);
		}
		strcpy(ap->option_flags,optflags); 
	}
	ap->vflag_cnt = (char) vflagcnt;		   
	ap->variant_param_cnt = (char) vparamcnt;
	if(vflagcnt) {
		if((ap->variant_list  = (char *)malloc((size_t)(vflagcnt+1)))==NULL) {
			sprintf(errstr,"INSUFFICIENT MEMORY: for variant_list\n");
			return(MEMORY_ERROR);
		}
		strcpy(ap->variant_list,varlist);		
		if((ap->variant_flags = (char *)malloc((size_t)(vflagcnt+1)))==NULL) {
			sprintf(errstr,"INSUFFICIENT MEMORY: for variant_flags\n");
			return(MEMORY_ERROR);
		}
		strcpy(ap->variant_flags,varflags);

	}
	return(FINISHED);
}

/***************************** APPLICATION_INIT **************************/

int application_init(dataptr dz)
{
	int exit_status;
	int storage_cnt;
	int tipc, brkcnt;
	aplptr ap = dz->application;
	if(ap->vflag_cnt>0)
		initialise_vflags(dz);	  
	tipc  = ap->max_param_cnt + ap->option_cnt + ap->variant_param_cnt;
	ap->total_input_param_cnt = (char)tipc;
	if(tipc>0) {
		if((exit_status = setup_input_param_range_stores(tipc,ap))<0)			  
			return(exit_status);
		if((exit_status = setup_input_param_defaultval_stores(tipc,ap))<0)		  
			return(exit_status);
		if((exit_status = setup_and_init_input_param_activity(dz,tipc))<0)	  
			return(exit_status);
	}
	brkcnt = tipc;
	//THERE ARE NO INPUTFILE brktables USED IN THIS PROCESS
	if(brkcnt>0) {
		if((exit_status = setup_and_init_input_brktable_constants(dz,brkcnt))<0)			  
			return(exit_status);
	}
	if((storage_cnt = tipc + ap->internal_param_cnt)>0) {		  
		if((exit_status = setup_parameter_storage_and_constants(storage_cnt,dz))<0)	  
			return(exit_status);
		if((exit_status = initialise_is_int_and_no_brk_constants(storage_cnt,dz))<0)	  
			return(exit_status);
	}													   
 	if((exit_status = mark_parameter_types(dz,ap))<0)	  
		return(exit_status);
	
	// establish_infile_constants() replaced by
	dz->infilecnt = 1;
	//establish_bufptrs_and_extra_buffers():
	return(FINISHED);
}

/********************** SETUP_PARAMETER_STORAGE_AND_CONSTANTS ********************/
/* RWD mallo changed to calloc; helps debug verison run as release! */

int setup_parameter_storage_and_constants(int storage_cnt,dataptr dz)
{
	if((dz->param       = (double *)calloc(storage_cnt, sizeof(double)))==NULL) {
		sprintf(errstr,"setup_parameter_storage_and_constants(): 1\n");
		return(MEMORY_ERROR);
	}
	if((dz->iparam      = (int    *)calloc(storage_cnt, sizeof(int)   ))==NULL) {
		sprintf(errstr,"setup_parameter_storage_and_constants(): 2\n");
		return(MEMORY_ERROR);
	}
	if((dz->is_int      = (char   *)calloc(storage_cnt, sizeof(char)))==NULL) {
		sprintf(errstr,"setup_parameter_storage_and_constants(): 3\n");
		return(MEMORY_ERROR);
	}
	if((dz->no_brk      = (char   *)calloc(storage_cnt, sizeof(char)))==NULL) {
		sprintf(errstr,"setup_parameter_storage_and_constants(): 5\n");
		return(MEMORY_ERROR);
	}
	return(FINISHED);
}

/************** INITIALISE_IS_INT_AND_NO_BRK_CONSTANTS *****************/

int initialise_is_int_and_no_brk_constants(int storage_cnt,dataptr dz)
{
	int n;
	for(n=0;n<storage_cnt;n++) {
		dz->is_int[n] = (char)0;
		dz->no_brk[n] = (char)0;
	}
	return(FINISHED);
}

/***************************** MARK_PARAMETER_TYPES **************************/

int mark_parameter_types(dataptr dz,aplptr ap)
{
	int n, m;							/* PARAMS */
	for(n=0;n<ap->max_param_cnt;n++) {
		switch(ap->param_list[n]) {
		case('0'):	break; /* dz->is_active[n] = 0 is default */
		case('i'):	dz->is_active[n] = (char)1; dz->is_int[n] = (char)1;dz->no_brk[n] = (char)1; break;
		case('I'):	dz->is_active[n] = (char)1;	dz->is_int[n] = (char)1; 						 break;
		case('d'):	dz->is_active[n] = (char)1;							dz->no_brk[n] = (char)1; break;
		case('D'):	dz->is_active[n] = (char)1;	/* normal case: double val or brkpnt file */	 break;
		default:
			sprintf(errstr,"Programming error: invalid parameter type in mark_parameter_types()\n");
			return(PROGRAM_ERROR);
		}
	}						 		/* OPTIONS */
	for(n=0,m=ap->max_param_cnt;n<ap->option_cnt;n++,m++) {
		switch(ap->option_list[n]) {
		case('i'): dz->is_active[m] = (char)1; dz->is_int[m] = (char)1;	dz->no_brk[m] = (char)1; break;
		case('I'): dz->is_active[m] = (char)1; dz->is_int[m] = (char)1; 						 break;
		case('d'): dz->is_active[m] = (char)1; 							dz->no_brk[m] = (char)1; break;
		case('D'): dz->is_active[m] = (char)1;	/* normal case: double val or brkpnt file */	 break;
		default:
			sprintf(errstr,"Programming error: invalid option type in mark_parameter_types()\n");
			return(PROGRAM_ERROR);
		}
	}								/* VARIANTS */
	for(n=0,m=ap->max_param_cnt + ap->option_cnt;n < ap->variant_param_cnt; n++, m++) {
		switch(ap->variant_list[n]) {
		case('0'): break;
		case('i'): dz->is_active[m] = (char)1; dz->is_int[m] = (char)1;	dz->no_brk[m] = (char)1; break;
		case('I'): dz->is_active[m] = (char)1; dz->is_int[m] = (char)1;	 						 break;
		case('d'): dz->is_active[m] = (char)1; 							dz->no_brk[m] = (char)1; break;
		case('D'): dz->is_active[m] = (char)1; /* normal case: double val or brkpnt file */		 break;
		default:
			sprintf(errstr,"Programming error: invalid variant type in mark_parameter_types()\n");
			return(PROGRAM_ERROR);
		}
	}								/* INTERNAL */
	for(n=0,
	m=ap->max_param_cnt + ap->option_cnt + ap->variant_param_cnt; n<ap->internal_param_cnt; n++,m++) {
		switch(ap->internal_param_list[n]) {
		case('0'):  break;	 /* dummy variables: variables not used: but important for internal paream numbering!! */
		case('i'):	dz->is_int[m] = (char)1;	dz->no_brk[m] = (char)1;	break;
		case('d'):								dz->no_brk[m] = (char)1;	break;
		default:
			sprintf(errstr,"Programming error: invalid internal param type in mark_parameter_types()\n");
			return(PROGRAM_ERROR);
		}
	}
	return(FINISHED);
}

/************************ HANDLE_THE_OUTFILE *********************/

int handle_the_outfile(int *cmdlinecnt,char ***cmdline,dataptr dz)
{
	int exit_status;
	char *filename = (*cmdline)[0];
	if(filename[0]=='-' && filename[1]=='f') {
		dz->floatsam_output = 1;
		dz->true_outfile_stype = SAMP_FLOAT;
		filename+= 2;
	}
	if(!sloom) {
		if(file_has_invalid_startchar(filename) || value_is_numeric(filename)) {
			sprintf(errstr,"Outfile name %s has invalid start character(s) or looks too much like a number.\n",filename);
			return(DATA_ERROR);
		}
	}
	strcpy(dz->outfilename,filename);	   
	if((exit_status = create_sized_outfile(filename,dz))<0)
		return(exit_status);
	(*cmdline)++;
	(*cmdlinecnt)--;
	return(FINISHED);
}

/***************************** ESTABLISH_APPLICATION **************************/

int establish_application(dataptr dz)
{
	aplptr ap;
	if((dz->application = (aplptr)malloc(sizeof (struct applic)))==NULL) {
		sprintf(errstr,"establish_application()\n");
		return(MEMORY_ERROR);
	}
	ap = dz->application;
	memset((char *)ap,0,sizeof(struct applic));
	return(FINISHED);
}

/************************* INITIALISE_VFLAGS *************************/

int initialise_vflags(dataptr dz)
{
	int n;
	if((dz->vflag  = (char *)malloc(dz->application->vflag_cnt * sizeof(char)))==NULL) {
		sprintf(errstr,"INSUFFICIENT MEMORY: vflag store,\n");
		return(MEMORY_ERROR);
	}
	for(n=0;n<dz->application->vflag_cnt;n++)
		dz->vflag[n]  = FALSE;
	return FINISHED;
}

/************************* SETUP_INPUT_PARAM_DEFAULTVALS *************************/

int setup_input_param_defaultval_stores(int tipc,aplptr ap)
{
	int n;
	if((ap->default_val = (double *)malloc(tipc * sizeof(double)))==NULL) {
		sprintf(errstr,"INSUFFICIENT MEMORY for application default values store\n");
		return(MEMORY_ERROR);
	}
	for(n=0;n<tipc;n++)
		ap->default_val[n] = 0.0;
	return(FINISHED);
}

/***************************** SETUP_AND_INIT_INPUT_PARAM_ACTIVITY **************************/

int setup_and_init_input_param_activity(dataptr dz,int tipc)
{
	int n;
	if((dz->is_active = (char   *)malloc((size_t)tipc))==NULL) {
		sprintf(errstr,"setup_and_init_input_param_activity()\n");
		return(MEMORY_ERROR);
	}
	for(n=0;n<tipc;n++)
		dz->is_active[n] = (char)0;
	return(FINISHED);
}

/************************* SETUP_HOVER_APPLICATION *******************/

int setup_hover_application(dataptr dz)
{
	int exit_status;
	aplptr ap;
	if((exit_status = establish_application(dz))<0)		// GLOBAL
		return(FAILED);
	ap = dz->application;
	// SEE parstruct FOR EXPLANATION of next 2 functions
	if((exit_status = set_param_data(ap,0   ,6,6,"DDDDdd"))<0)
		return(FAILED);
	if((exit_status = set_vflgs(ap,"",0,"","",0,0,""))<0)
		return(FAILED);
	// set_legal_infile_structure -->
	dz->has_otherfile = FALSE;
	// assign_process_logic -->
	dz->input_data_type = SNDFILES_ONLY;
	dz->process_type	= UNEQUAL_SNDFILE;	
	dz->outfiletype  	= SNDFILE_OUT;
	return application_init(dz);	//GLOBAL
}

/************************* PARSE_INFILE_AND_CHECK_TYPE *******************/

int parse_infile_and_check_type(char **cmdline,dataptr dz)
{
	int exit_status;
	infileptr infile_info;
	if(!sloom) {
		if((infile_info = (infileptr)malloc(sizeof(struct filedata)))==NULL) {
			sprintf(errstr,"INSUFFICIENT MEMORY for infile structure to test file data.");
			return(MEMORY_ERROR);
		} else if((exit_status = cdparse(cmdline[0],infile_info))<0) {
			sprintf(errstr,"Failed to parse input file %s\n",cmdline[0]);
			return(PROGRAM_ERROR);
		} else if(infile_info->filetype != SNDFILE)  {
			sprintf(errstr,"File %s is not of correct type\n",cmdline[0]);
			return(DATA_ERROR);
		} else if(infile_info->channels != 1)  {
			sprintf(errstr,"File %s is not of correct type (must be mono)\n",cmdline[0]);
			return(DATA_ERROR);
		} else if((exit_status = copy_parse_info_to_main_structure(infile_info,dz))<0) {
			sprintf(errstr,"Failed to copy file parsing information\n");
			return(PROGRAM_ERROR);
		}
		free(infile_info);
	}
	return(FINISHED);
}

/************************* SETUP_HOVER_PARAM_RANGES_AND_DEFAULTS *******************/

int setup_hover_param_ranges_and_defaults(dataptr dz)
{
	int exit_status;
	aplptr ap = dz->application;
	// set_param_ranges()
	ap->total_input_param_cnt = (char)(ap->max_param_cnt + ap->option_cnt + ap->variant_param_cnt);
	// NB total_input_param_cnt is > 0 !!!
	if((exit_status = setup_input_param_range_stores(ap->total_input_param_cnt,ap))<0)
		return(FAILED);
	// get_param_ranges()
	ap->lo[0]	= 1.0 / (dz->duration * 2.0);
	ap->hi[0]	= dz->nyquist;
	ap->default_val[0]	= 440.0;
	ap->lo[1]	= 0.0;
	ap->hi[1]	= dz->duration;
	ap->default_val[1] = dz->duration / 2.0;
	ap->lo[2]	= 0.0;
	ap->hi[2]	= 1.0;
	ap->default_val[2] = 0.1;
	ap->lo[3]	= 0.0;
	ap->hi[3]	= 1.0;
	ap->default_val[3] = 0.1;
	ap->lo[4]	= 0.0;
	ap->hi[4]	= 100.0;
	ap->default_val[4] = 1.0;
	ap->lo[5]	= 0.0;
	ap->hi[5]	= 32767.0;
	ap->default_val[5] = 10.0;

	dz->maxmode = 0;
	if(!sloom)
		put_default_vals_in_all_params(dz);
	return(FINISHED);
}

/********************************* PARSE_SLOOM_DATA *********************************/

int parse_sloom_data(int argc,char *argv[],char ***cmdline,int *cmdlinecnt,dataptr dz)
{
	int exit_status;
	int cnt = 1, infilecnt;
	int filesize, insams, inbrksize;
	double dummy;
	int true_cnt = 0;
	aplptr ap;

	while(cnt<=PRE_CMDLINE_DATACNT) {
		if(cnt > argc) {
			sprintf(errstr,"Insufficient data sent from TK\n");
			return(DATA_ERROR);
		}
		switch(cnt) {
		case(1):	
			if(sscanf(argv[cnt],"%d",&dz->process)!=1) {
				sprintf(errstr,"Cannot read process no. sent from TK\n");
				return(DATA_ERROR);
			}
			break;

		case(2):	
			if(sscanf(argv[cnt],"%d",&dz->mode)!=1) {
				sprintf(errstr,"Cannot read mode no. sent from TK\n");
				return(DATA_ERROR);
			}
			if(dz->mode > 0)
				dz->mode--;
			//setup_particular_application() =
			if((exit_status = setup_hover_application(dz))<0)
				return(exit_status);
			ap = dz->application;
			break;

		case(3):	
			if(sscanf(argv[cnt],"%d",&infilecnt)!=1) {
				sprintf(errstr,"Cannot read infilecnt sent from TK\n");
				return(DATA_ERROR);
			}
			if(infilecnt < 1) {
				true_cnt = cnt + 1;
				cnt = PRE_CMDLINE_DATACNT;	/* force exit from loop after assign_file_data_storage */
			}
			if((exit_status = assign_file_data_storage(infilecnt,dz))<0)
				return(exit_status);
			break;
		case(INPUT_FILETYPE+4):	
			if(sscanf(argv[cnt],"%d",&dz->infile->filetype)!=1) {
				sprintf(errstr,"Cannot read filetype sent from TK (%s)\n",argv[cnt]);
				return(DATA_ERROR);
			}
			break;
		case(INPUT_FILESIZE+4):	
			if(sscanf(argv[cnt],"%d",&filesize)!=1) {
				sprintf(errstr,"Cannot read infilesize sent from TK\n");
				return(DATA_ERROR);
			}
			dz->insams[0] = filesize;	
			break;
		case(INPUT_INSAMS+4):	
			if(sscanf(argv[cnt],"%d",&insams)!=1) {
				sprintf(errstr,"Cannot read insams sent from TK\n");
				return(DATA_ERROR);
			}
			dz->insams[0] = insams;	
			break;
		case(INPUT_SRATE+4):	
			if(sscanf(argv[cnt],"%d",&dz->infile->srate)!=1) {
				sprintf(errstr,"Cannot read srate sent from TK\n");
				return(DATA_ERROR);
			}
			break;
		case(INPUT_CHANNELS+4):	
			if(sscanf(argv[cnt],"%d",&dz->infile->channels)!=1) {
				sprintf(errstr,"Cannot read channels sent from TK\n");
				return(DATA_ERROR);
			}
			break;
		case(INPUT_STYPE+4):	
			if(sscanf(argv[cnt],"%d",&dz->infile->stype)!=1) {
				sprintf(errstr,"Cannot read stype sent from TK\n");
				return(DATA_ERROR);
			}
			break;
		case(INPUT_ORIGSTYPE+4):	
			if(sscanf(argv[cnt],"%d",&dz->infile->origstype)!=1) {
				sprintf(errstr,"Cannot read origstype sent from TK\n");
				return(DATA_ERROR);
			}
			break;
		case(INPUT_ORIGRATE+4):	
			if(sscanf(argv[cnt],"%d",&dz->infile->origrate)!=1) {
				sprintf(errstr,"Cannot read origrate sent from TK\n");
				return(DATA_ERROR);
			}
			break;
		case(INPUT_MLEN+4):	
			if(sscanf(argv[cnt],"%d",&dz->infile->Mlen)!=1) {
				sprintf(errstr,"Cannot read Mlen sent from TK\n");
				return(DATA_ERROR);
			}
			break;
		case(INPUT_DFAC+4):	
			if(sscanf(argv[cnt],"%d",&dz->infile->Dfac)!=1) {
				sprintf(errstr,"Cannot read Dfac sent from TK\n");
				return(DATA_ERROR);
			}
			break;
		case(INPUT_ORIGCHANS+4):	
			if(sscanf(argv[cnt],"%d",&dz->infile->origchans)!=1) {
				sprintf(errstr,"Cannot read origchans sent from TK\n");
				return(DATA_ERROR);
			}
			break;
		case(INPUT_SPECENVCNT+4):	
			if(sscanf(argv[cnt],"%d",&dz->infile->specenvcnt)!=1) {
				sprintf(errstr,"Cannot read specenvcnt sent from TK\n");
				return(DATA_ERROR);
			}
			dz->specenvcnt = dz->infile->specenvcnt;
			break;
		case(INPUT_WANTED+4):	
			if(sscanf(argv[cnt],"%d",&dz->wanted)!=1) {
				sprintf(errstr,"Cannot read wanted sent from TK\n");
				return(DATA_ERROR);
			}
			break;
		case(INPUT_WLENGTH+4):	
			if(sscanf(argv[cnt],"%d",&dz->wlength)!=1) {
				sprintf(errstr,"Cannot read wlength sent from TK\n");
				return(DATA_ERROR);
			}
			break;
		case(INPUT_OUT_CHANS+4):	
			if(sscanf(argv[cnt],"%d",&dz->out_chans)!=1) {
				sprintf(errstr,"Cannot read out_chans sent from TK\n");
				return(DATA_ERROR);
			}
			break;
			/* RWD these chanegs to samps - tk will have to deal with that! */
		case(INPUT_DESCRIPTOR_BYTES+4):	
			if(sscanf(argv[cnt],"%d",&dz->descriptor_samps)!=1) {
				sprintf(errstr,"Cannot read descriptor_samps sent from TK\n");
				return(DATA_ERROR);
			}
			break;
		case(INPUT_IS_TRANSPOS+4):	
			if(sscanf(argv[cnt],"%d",&dz->is_transpos)!=1) {
				sprintf(errstr,"Cannot read is_transpos sent from TK\n");
				return(DATA_ERROR);
			}
			break;
		case(INPUT_COULD_BE_TRANSPOS+4):	
			if(sscanf(argv[cnt],"%d",&dz->could_be_transpos)!=1) {
				sprintf(errstr,"Cannot read could_be_transpos sent from TK\n");
				return(DATA_ERROR);
			}
			break;
		case(INPUT_COULD_BE_PITCH+4):	
			if(sscanf(argv[cnt],"%d",&dz->could_be_pitch)!=1) {
				sprintf(errstr,"Cannot read could_be_pitch sent from TK\n");
				return(DATA_ERROR);
			}
			break;
		case(INPUT_DIFFERENT_SRATES+4):	
			if(sscanf(argv[cnt],"%d",&dz->different_srates)!=1) {
				sprintf(errstr,"Cannot read different_srates sent from TK\n");
				return(DATA_ERROR);
			}
			break;
		case(INPUT_DUPLICATE_SNDS+4):	
			if(sscanf(argv[cnt],"%d",&dz->duplicate_snds)!=1) {
				sprintf(errstr,"Cannot read duplicate_snds sent from TK\n");
				return(DATA_ERROR);
			}
			break;
		case(INPUT_BRKSIZE+4):	
			if(sscanf(argv[cnt],"%d",&inbrksize)!=1) {
				sprintf(errstr,"Cannot read brksize sent from TK\n");
				return(DATA_ERROR);
			}
			if(inbrksize > 0) {
				switch(dz->input_data_type) {
				case(WORDLIST_ONLY):
					break;
				case(PITCH_AND_PITCH):
				case(PITCH_AND_TRANSPOS):
				case(TRANSPOS_AND_TRANSPOS):
					dz->tempsize = inbrksize;
					break;
				case(BRKFILES_ONLY):
				case(UNRANGED_BRKFILE_ONLY):
				case(DB_BRKFILES_ONLY):
				case(ALL_FILES):
				case(ANY_NUMBER_OF_ANY_FILES):
					if(dz->extrabrkno < 0) {
						sprintf(errstr,"Storage location number for brktable not established by CDP.\n");
						return(DATA_ERROR);
					}
					if(dz->brksize == NULL) {
						sprintf(errstr,"CDP has not established storage space for input brktable.\n");
						return(PROGRAM_ERROR);
					}
					dz->brksize[dz->extrabrkno]	= inbrksize;
					break;
				default:
					sprintf(errstr,"TK sent brktablesize > 0 for input_data_type [%d] not using brktables.\n",
					dz->input_data_type);
					return(PROGRAM_ERROR);
				}
				break;
			}
			break;
		case(INPUT_NUMSIZE+4):	
			if(sscanf(argv[cnt],"%d",&dz->numsize)!=1) {
				sprintf(errstr,"Cannot read numsize sent from TK\n");
				return(DATA_ERROR);
			}
			break;
		case(INPUT_LINECNT+4):	
			if(sscanf(argv[cnt],"%d",&dz->linecnt)!=1) {
				sprintf(errstr,"Cannot read linecnt sent from TK\n");
				return(DATA_ERROR);
			}
			break;
		case(INPUT_ALL_WORDS+4):	
			if(sscanf(argv[cnt],"%d",&dz->all_words)!=1) {
				sprintf(errstr,"Cannot read all_words sent from TK\n");
				return(DATA_ERROR);
			}
			break;
		case(INPUT_ARATE+4):	
			if(sscanf(argv[cnt],"%f",&dz->infile->arate)!=1) {
				sprintf(errstr,"Cannot read arate sent from TK\n");
				return(DATA_ERROR);
			}
			break;
		case(INPUT_FRAMETIME+4):	
			if(sscanf(argv[cnt],"%lf",&dummy)!=1) {
				sprintf(errstr,"Cannot read frametime sent from TK\n");
				return(DATA_ERROR);
			}
			dz->frametime = (float)dummy;
			break;
		case(INPUT_WINDOW_SIZE+4):	
			if(sscanf(argv[cnt],"%f",&dz->infile->window_size)!=1) {
				sprintf(errstr,"Cannot read window_size sent from TK\n");
					return(DATA_ERROR);
			}
			break;
		case(INPUT_NYQUIST+4):	
			if(sscanf(argv[cnt],"%lf",&dz->nyquist)!=1) {
				sprintf(errstr,"Cannot read nyquist sent from TK\n");
				return(DATA_ERROR);
			}
			break;
		case(INPUT_DURATION+4):	
			if(sscanf(argv[cnt],"%lf",&dz->duration)!=1) {
				sprintf(errstr,"Cannot read duration sent from TK\n");
				return(DATA_ERROR);
			}
			break;
		case(INPUT_MINBRK+4):	
			if(sscanf(argv[cnt],"%lf",&dz->minbrk)!=1) {
				sprintf(errstr,"Cannot read minbrk sent from TK\n");
				return(DATA_ERROR);
			}
			break;
		case(INPUT_MAXBRK+4):	
			if(sscanf(argv[cnt],"%lf",&dz->maxbrk)!=1) {
				sprintf(errstr,"Cannot read maxbrk sent from TK\n");
				return(DATA_ERROR);
			}
			break;
		case(INPUT_MINNUM+4):	
			if(sscanf(argv[cnt],"%lf",&dz->minnum)!=1) {
				sprintf(errstr,"Cannot read minnum sent from TK\n");
				return(DATA_ERROR);
			}
			break;
		case(INPUT_MAXNUM+4):	
			if(sscanf(argv[cnt],"%lf",&dz->maxnum)!=1) {
				sprintf(errstr,"Cannot read maxnum sent from TK\n");
				return(DATA_ERROR);
			}
			break;
		default:
			sprintf(errstr,"case switch item missing: parse_sloom_data()\n");
			return(PROGRAM_ERROR);
		}
		cnt++;
	}
	if(cnt!=PRE_CMDLINE_DATACNT+1) {
		sprintf(errstr,"Insufficient pre-cmdline params sent from TK\n");
		return(DATA_ERROR);
	}

	if(true_cnt)
		cnt = true_cnt;
	*cmdlinecnt = 0;		

	while(cnt < argc) {
		if((exit_status = get_tk_cmdline_word(cmdlinecnt,cmdline,argv[cnt]))<0)
			return(exit_status);
		cnt++;
	}
	return(FINISHED);
}

/********************************* GET_TK_CMDLINE_WORD *********************************/

int get_tk_cmdline_word(int *cmdlinecnt,char ***cmdline,char *q)
{
	if(*cmdlinecnt==0) {
		if((*cmdline = (char **)malloc(sizeof(char *)))==NULL)	{
			sprintf(errstr,"INSUFFICIENT MEMORY for TK cmdline array.\n");
			return(MEMORY_ERROR);
		}
	} else {
		if((*cmdline = (char **)realloc(*cmdline,((*cmdlinecnt)+1) * sizeof(char *)))==NULL)	{
			sprintf(errstr,"INSUFFICIENT MEMORY for TK cmdline array.\n");
			return(MEMORY_ERROR);
		}
	}
	if(((*cmdline)[*cmdlinecnt] = (char *)malloc((strlen(q) + 1) * sizeof(char)))==NULL)	{
		sprintf(errstr,"INSUFFICIENT MEMORY for TK cmdline item %d.\n",(*cmdlinecnt)+1);
		return(MEMORY_ERROR);
	}
	strcpy((*cmdline)[*cmdlinecnt],q);
	(*cmdlinecnt)++;
	return(FINISHED);
}


/****************************** ASSIGN_FILE_DATA_STORAGE *********************************/

int assign_file_data_storage(int infilecnt,dataptr dz)
{
	int exit_status;
	int no_sndfile_system_files = FALSE;
	dz->infilecnt = infilecnt;
	if((exit_status = allocate_filespace(dz))<0)
		return(exit_status);
	if(no_sndfile_system_files)
		dz->infilecnt = 0;
	return(FINISHED);
}



/************************* redundant functions: to ensure libs compile OK *******************/

int assign_process_logic(dataptr dz)
{
	return(FINISHED);
}

void set_legal_infile_structure(dataptr dz)
{}

int set_legal_internalparam_structure(int process,int mode,aplptr ap)
{
	return(FINISHED);
}

int setup_internal_arrays_and_array_pointers(dataptr dz)
{
	return(FINISHED);
}

int establish_bufptrs_and_extra_buffers(dataptr dz)
{
	return(FINISHED);
}

int read_special_data(char *str,dataptr dz)	
{
	return(FINISHED);
}

int inner_loop
(int *peakscore,int *descnt,int *in_start_portion,int *least,int *pitchcnt,int windows_in_buf,dataptr dz)
{
	return(FINISHED);
}

int get_process_no(char *prog_identifier_from_cmdline,dataptr dz)
{
	return(FINISHED);
}


/******************************** USAGE1 ********************************/

int usage1(void)
{
	usage2("hover");
	return(USAGE_ONLY);
}

/**************************** CHECK_HOVER_PARAM_VALIDITY_AND_CONSISTENCY *****************************/

int check_hover_param_validity_and_consistency(dataptr dz)
{
	int exit_status;
	int mintraverse;
	double maxval, splic;
	if(dz->brksize[HOVER_FRQ] == 0) {
					/* TRAVERSE = TOTAL zig+zag read, in samples */
		dz->iparam[TRAVERSE] = (int)round(dz->infile->srate/dz->param[HOVER_FRQ]);
		if(dz->iparam[TRAVERSE] < 4)
			dz->iparam[TRAVERSE] = 4;
		if(dz->iparam[TRAVERSE] > dz->insams[0] * 2)
			dz->iparam[TRAVERSE] = dz->insams[0] * 2;
		mintraverse = dz->iparam[TRAVERSE];
	} else {
		if((exit_status = get_maxvalue_in_brktable(&maxval,0,dz)) < 0)
			return(exit_status);
		mintraverse = (int)round(dz->infile->srate/maxval);
		if(mintraverse < 4)
			mintraverse = 4;
		if(mintraverse > dz->insams[0] * 2)
			mintraverse = dz->insams[0] * 2;
	}
	dz->iparam[HOVER_SPLIC] = (int)round(dz->param[HOVER_SPLIC] * MS_TO_SECS * dz->infile->srate);
	if(dz->iparam[HOVER_SPLIC] * 2 >= mintraverse) {
		splic = (double)(mintraverse/2);
		splic /= (double)dz->infile->srate;
		splic *= SECS_TO_MS;
		sprintf(errstr,"Splicelen incompatible with maximum frequency: must be less than %lf mS\n",splic);
		return(DATA_ERROR);
	}
	if(dz->brksize[HOVER_LOC] == 0)
		dz->iparam[HOVER_LOC] = (int)round(dz->infile->srate * dz->param[HOVER_LOC]);
	dz->iparam[HOVER_DUR] = (int)round(dz->infile->srate * dz->param[HOVER_DUR]);
	return FINISHED;
}

/********************************************************************************************/

int get_the_process_no(char *prog_identifier_from_cmdline,dataptr dz)
{
	if(!strcmp(prog_identifier_from_cmdline,"hover"))				dz->process = HOVER;
	else {
		sprintf(errstr,"Unknown program identification string '%s'\n",prog_identifier_from_cmdline);
		return(USAGE_ONLY);
	}
	return(FINISHED);
}

/******************************** SETUP_AND_INIT_INPUT_BRKTABLE_CONSTANTS ********************************/

int setup_and_init_input_brktable_constants(dataptr dz,int brkcnt)
{	
	int n;
	if((dz->brk      = (double **)malloc(brkcnt * sizeof(double *)))==NULL) {
		sprintf(errstr,"setup_and_init_input_brktable_constants(): 1\n");
		return(MEMORY_ERROR);
	}
	if((dz->brkptr   = (double **)malloc(brkcnt * sizeof(double *)))==NULL) {
		sprintf(errstr,"setup_and_init_input_brktable_constants(): 6\n");
		return(MEMORY_ERROR);
	}
	if((dz->brksize  = (int    *)malloc(brkcnt * sizeof(int)))==NULL) {
		sprintf(errstr,"setup_and_init_input_brktable_constants(): 2\n");
		return(MEMORY_ERROR);
	}
	if((dz->firstval = (double  *)malloc(brkcnt * sizeof(double)))==NULL) {
		sprintf(errstr,"setup_and_init_input_brktable_constants(): 3\n");
		return(MEMORY_ERROR);												  
	}
	if((dz->lastind  = (double  *)malloc(brkcnt * sizeof(double)))==NULL) {
		sprintf(errstr,"setup_and_init_input_brktable_constants(): 4\n");
		return(MEMORY_ERROR);
	}
	if((dz->lastval  = (double  *)malloc(brkcnt * sizeof(double)))==NULL) {
		sprintf(errstr,"setup_and_init_input_brktable_constants(): 5\n");
		return(MEMORY_ERROR);
	}
	if((dz->brkinit  = (int     *)malloc(brkcnt * sizeof(int)))==NULL) {
		sprintf(errstr,"setup_and_init_input_brktable_constants(): 7\n");
		return(MEMORY_ERROR);
	}
	for(n=0;n<brkcnt;n++) {
		dz->brk[n]     = NULL;
		dz->brkptr[n]  = NULL;
		dz->brkinit[n] = 0;
		dz->brksize[n] = 0;
	}
	return(FINISHED);
}

/******************************** USAGE2 ********************************/

int usage2(char *str)
{
	if(!strcmp(str,"hover")) {
		fprintf(stderr,
	    "USAGE:\n"
	    "hover hover infile outfile frq loc frqrand locrand splice dur\n"
		"\n"
		"move through a file, zigzag reading it at a given frequency.\n"
		"\n"
		"FRQ      rate of reading source-samples (in Hz).\n"
		"         determines samplewidth of zigzag-read. e.g. at srate 44100,\n"
		"         frq 1 Hz, reads 22050 samps forward, & 22050 samps back.\n"
		"         frq 10 Hz, reads 2205 samps forward, & 2205 samps back.\n"
		"LOC      Time in sourcefile from which samples are read.\n"
		"FRQRAND  Random variation of frequency, (0-1).\n"
		"LOCRAND  Random variation of location, (0-1).\n"
		"SPLICE   length of splices at zig (zag) ends (in mS).\n"
		"DUR      total output duration.\n"
		"\n"
		"frq and loc (frqrand and locrand) may vary through time.\n"
		"time in any brkpoint files is time in output-file.\n"
		"splicelen must be less than 1 over twice the max frq used.\n"
		"\n");
	} else
		fprintf(stdout,"Unknown option '%s'\n",str);
	return(USAGE_ONLY);
}

int usage3(char *str1,char *str2)
{
	fprintf(stderr,"Insufficient parameters on command line.\n");
	return(USAGE_ONLY);
}

/******************************** HOVER ********************************/

int hover(dataptr dz)
{
	int exit_status;
	double randvar, time, val;
	int samptime, next_samptime, randvarsamps;
	float *ibuf = dz->sampbuf[0];
	float *obuf = dz->sampbuf[1];
	float *splicebuf = dz->sampbuf[3];
	int ibufpos, obufpos, abs_startreadbuf, abs_endreadbuf, readlimit;
	int traverse, max_traverse, location, next_traverse, next_max_traverse, next_location, quarter_cycle;
	int step, zig, zag, maxzag, endplicestart, abs_samp_position;
	int splbufpos, n, k;
	double splice, splincr;

	dz->itemcnt = 0;		/* counts signal offsets from centre */
	dz->peak_fval = 1.0;	/* used to store lastval for the offset calculations */
	
	obufpos = 0;
	splincr = 1.0/(double)dz->iparam[HOVER_SPLIC];

	memset((char *)obuf,0,dz->buflen * 2 * sizeof(float));
	memset((char *)splicebuf,0,dz->buflen * sizeof(float));
	if((exit_status = read_samps(ibuf,dz))<0)
		return(exit_status);
	next_samptime = 0;
	time = 0.0;
	abs_startreadbuf = 0;
	abs_endreadbuf = dz->ssampsread;

	/* No point in attempting to read buffers too close to end of file */	

	if(dz->insams[0] < dz->buflen)
		readlimit = 0;
	else {
		readlimit = (dz->insams[0] - dz->buflen)/F_SECSIZE;
		readlimit *= F_SECSIZE;
		if(dz->insams[0] - readlimit > dz->buflen)
			readlimit += F_SECSIZE;
	}

	/* Initialise process by finding first frq, first location, (including randomisings) */

	if((exit_status = read_values_from_all_existing_brktables(time,dz))<0)
	return(exit_status);
	
	/* TRAVERSE = TOTAL zig+zag read, in samples */

	if(dz->brksize[HOVER_FRQ] > 0)
		dz->iparam[TRAVERSE] = (int)round(dz->infile->srate/dz->param[HOVER_FRQ]);
	next_traverse = dz->iparam[TRAVERSE];

			/* LOCATION */

	if(dz->brksize[HOVER_LOC] > 0)
		dz->iparam[HOVER_LOC] = (int)round(dz->infile->srate * dz->param[HOVER_LOC]);
	next_location = dz->iparam[HOVER_LOC];

		/* RANDVAR of LOCATION */

	if(dz->param[HOVER_LOCR] > 0.0) {
		randvar = dz->param[HOVER_LOCR] * drand48(); /* range 0 to +HOVER_LOCR */
		randvar *= 2.0;
		randvar -= 1.0;	/* range -HOVER_LOCR to +HOVER_LOCR */
			/* random variation in location is random relative width of current hover-traverse */
			/* rather than random relative to the total file duration */
		randvarsamps = (int)round(next_traverse * randvar);
		next_location += randvarsamps;
	}

	/* TRAVERSE, RAND VARIATION */

	if(dz->param[HOVER_FRQR] > 0.0) {
		randvar = dz->param[HOVER_FRQR] * drand48(); /* range 0 to +HOVER_FRQR */
		randvar *= 2.0;
		randvar -= 1.0;	/* range -HOVER_FRQR to +HOVER_FRQR */
		randvarsamps = (int)round(next_traverse * randvar);
		next_traverse += randvarsamps;
	}

	if(next_traverse <= dz->iparam[HOVER_SPLIC] * 2)
		next_traverse = (dz->iparam[HOVER_SPLIC] * 2) + 1;
	if(next_traverse > dz->insams[0] * 2)
		next_traverse = dz->insams[0] * 2;

		/* The zigzagging actually starts to the 'left' of the location */
		/* and we want to do our calculations from this startpoint, so */

	quarter_cycle = (int)round((double)next_traverse/4.0);
	next_location -= quarter_cycle;
	
		/* CHECK LOCATION IS NOT OUT OF BOUNDS */

	if(next_location < 0)
		next_location = 0;
	if(next_location >= dz->insams[0])
		next_location = dz->insams[0] - 1;

		/* CHECK TRAVERSE IS NOT OUT OF BOUNDS */

	next_max_traverse = (2 * dz->insams[0]) - next_location;
	if(next_traverse > next_max_traverse)
		next_traverse = next_max_traverse;

		/* ADVANCE TO NEXT LOCATION - we need to know if the hover will drift from current location */
		/* BEFORE we create the zig-zagging motion */
		/* SO WE'RE ALWAYS CALCULATING THE DATA, 1 step ahead of where we're creating the zigzag */

	samptime = next_samptime;

		/* Next time (in outfile) is after traverse-samples are written, */
		/* and out-pointer has backtracked for splice-overlap */

	next_samptime += (next_traverse - dz->iparam[HOVER_SPLIC]);

	while(samptime < dz->iparam[HOVER_DUR]) {
		
		traverse = next_traverse;
		location = next_location;
		max_traverse = next_max_traverse;

			/* we need to know where next location is, in order to know whether */
			/* hover drifts forward, backwards, or stays centred roughly where it is */
			/* so we may as well calculate EVERYTHING about the next location here */

			/* CALCULATE NEXT LOCATION (and TRAVERSE) */

		time = (double)next_samptime/(double)dz->infile->srate;
		if((exit_status = read_values_from_all_existing_brktables(time,dz))<0)
			return(exit_status);

		if(dz->brksize[HOVER_FRQ] > 0)
			dz->iparam[TRAVERSE] = (int)round(dz->infile->srate/dz->param[HOVER_FRQ]);
		next_traverse = dz->iparam[TRAVERSE];

		if(dz->brksize[HOVER_LOC] > 0)
			dz->iparam[HOVER_LOC] = (int)round(dz->infile->srate * dz->param[HOVER_LOC]);
		next_location = dz->iparam[HOVER_LOC];

		if(dz->param[HOVER_LOCR] > 0.0) {
			randvar = dz->param[HOVER_LOCR] * drand48();
			randvar *= 2.0;
			randvar -= 1.0;
			randvarsamps = (int)round(next_traverse * randvar);
			next_location += randvarsamps;
		}
		if(dz->param[HOVER_FRQR] > 0.0) {
			randvar = dz->param[HOVER_FRQR] * drand48();
			randvar *= 2.0;
			randvar -= 1.0;
			randvarsamps = (int)round(next_traverse * randvar);
			next_traverse += randvarsamps;
		}
		if(next_traverse <= dz->iparam[HOVER_SPLIC] * 2)
			next_traverse = (dz->iparam[HOVER_SPLIC] * 2) + 1;
		if(next_traverse > dz->insams[0] * 2)
			next_traverse = dz->insams[0] * 2;

		quarter_cycle = (int)round((double)next_traverse/4.0);
		next_location -= quarter_cycle;

		if(next_location < 0)
			next_location = 0;
		if(next_location >= dz->insams[0])
			next_location = dz->insams[0] - 1;

		next_max_traverse = (2 * dz->insams[0]) - next_location;
		if(next_traverse > next_max_traverse)
			next_traverse = next_max_traverse;

				/* go get input samples at Hover Location and setup ibuf pointer */

		if(location >= abs_endreadbuf || location < abs_startreadbuf) {
			abs_startreadbuf = location/F_SECSIZE;
			abs_startreadbuf *= F_SECSIZE;	/* align new read with sector size */
			if(abs_startreadbuf > readlimit)
				abs_startreadbuf = readlimit;
			if((sndseekEx(dz->ifd[0],abs_startreadbuf,0)<0)){
                sprintf(errstr,"sndseek() failed\n");
                return SYSTEM_ERROR;
            }
			if((exit_status = read_samps(ibuf,dz))<0)
				return exit_status;
			abs_endreadbuf = abs_startreadbuf + dz->ssampsread;
		}
		ibufpos = location - abs_startreadbuf;

			/* does hover drift forwards or backwards ?? calculate appropriate zig and zag */

		step = next_location - location;
		if(abs(step) > traverse) {
			if(step > 0)
				zig = traverse;				/* if +ve step exceeds traverse, just go forward by complete traverse */
			else
				zig = 0;					/* if -ve step exceeds traverse, just go backwards by complete traverse */
		} else {
			zig = (step + traverse)/2;		/* amazingly in all cases! */
		}
		if((zig + location) > dz->insams[0])	/* check for end overshoot */
			zig = dz->insams[0] - location;
		
		zag = traverse - zig;
			/* it should be impossible for zag to overshoot, but check it anyway */
		maxzag = max_traverse - zig;
		if(zag > maxzag) {
			zag = maxzag;
			traverse = zig + zag;	
		}
			/* copy zig and zag to splicebuf */
		splbufpos = 0;
		splice = splincr;
		endplicestart = traverse - dz->iparam[HOVER_SPLIC];
		for(k=0,n=0;n<zig;n++,k++) {
			if(ibufpos >= dz->ssampsread) {
				if((exit_status = read_samps(ibuf,dz))<0)
					return exit_status;
				if(dz->ssampsread == 0) {
					sprintf(errstr,"Problem in read_buffer accounting.\n");
					return(PROGRAM_ERROR);
				}
				abs_startreadbuf += dz->buflen;
				abs_endreadbuf = abs_startreadbuf + dz->ssampsread;
				ibufpos = 0;
			}
			val = ibuf[ibufpos++];
			if(k < dz->iparam[HOVER_SPLIC]) {
				val *= splice;
				splice += splincr;
			} else if(k >= endplicestart) {
				splice -= splincr;
				val *= splice;
			}
			splicebuf[splbufpos++] = (float)val;
			if(splbufpos >= dz->buflen) {
				if((exit_status = copy_to_output(&obufpos,dz->buflen,dz)) < 0)
					return(exit_status);
				splbufpos = 0;
			}
		}
		for(n=0;n<zag;n++,k++) {
			if(ibufpos < 0) {
				abs_samp_position = abs_startreadbuf;
				abs_startreadbuf -= dz->buflen;
				if(abs_startreadbuf < 0) {
					abs_startreadbuf = 0;			// Set to read from start of file 
					ibufpos = abs_samp_position;	// position of next read is exactly where we'd got to.
				} else {							// Otherwise we get full buffer adjacent to last,
					ibufpos = dz->buflen - 1;		// and start read at its end
				}
				if((sndseekEx(dz->ifd[0],abs_startreadbuf,0)<0)){
                    sprintf(errstr,"sndseek() failed\n");
                    return SYSTEM_ERROR;
                }
				if((exit_status = read_samps(ibuf,dz))<0)
					return exit_status;
				abs_endreadbuf = abs_startreadbuf + dz->ssampsread;
			}
			val = ibuf[ibufpos--];
			if(k < dz->iparam[HOVER_SPLIC]) {
				val *= splice;
				splice += splincr;
			} else if(k >= endplicestart) {
				splice -= splincr;
				val *= splice;
			}
			splicebuf[splbufpos++] = (float)val;
			if(splbufpos >= dz->buflen) {
				if((exit_status = copy_to_output(&obufpos,dz->buflen,dz)) < 0)
					return(exit_status);
				splbufpos = 0;
			}
		}
		if(splbufpos > 0) {
			if((exit_status = copy_to_output(&obufpos,splbufpos,dz)) < 0)
				return(exit_status);
		}
		
				/* back track in outbuf to do splice-overlap for next chunk of spliced data */

		obufpos -= dz->iparam[HOVER_SPLIC];
				
				/* advance time (in output) */

		samptime = next_samptime;
		next_samptime += (next_traverse - dz->iparam[HOVER_SPLIC]);
	}
				/* write any samples still left in output buffer */

	if(obufpos > 0) {
		if((exit_status = write_samps(obuf,obufpos,dz))<0)
			return(exit_status);
	}
	return FINISHED;
}	

/******************************** COPY_TO_OUTPUT ********************************/

int copy_to_output(int *obufpos,int sampcnt,dataptr dz)
{
	int exit_status;
	float *obuf		 = dz->sampbuf[1];
	float *overflow  = dz->sampbuf[2];
	float *splicebuf = dz->sampbuf[3];
	int  n, opos = *obufpos;
	int samps_to_copy, write_limit = dz->buflen + dz->iparam[HOVER_SPLIC];
	double val, loc;
	for(n=0;n<sampcnt;n++) {
		val = obuf[opos] + splicebuf[n];
		if(dz->itemcnt >= 0) {
			if(dz->peak_fval > 0.0) {
				if(val > 0.0)
					dz->itemcnt++;
				else
					dz->itemcnt = 0;
			} else if(dz->peak_fval < 0.0) {
				if(val < 0.0)
					dz->itemcnt++;
				else
					dz->itemcnt = 0;
			}
			dz->peak_fval = val;
			if(dz->itemcnt >= dz->nyquist) {
				loc = (double)(dz->total_samps_written + opos)/(double)(dz->infile->srate);
				fprintf(stdout,"WARNING: At least one area of off-centre signal detected (at %lf secs)\n",loc);
				fflush(stdout);
				dz->itemcnt = -1;
			}
		}
		obuf[opos++] = (float)val;
		if(opos >= write_limit) {
			if((exit_status = write_samps(obuf,dz->buflen,dz))<0)
				return(exit_status);
			memset((char *)obuf,0,dz->buflen * sizeof(float));
			samps_to_copy = write_limit - dz->buflen;
			memcpy((char *)obuf,(char *)overflow,samps_to_copy * sizeof(float));
			memset((char *)overflow,0,dz->buflen * sizeof(float));
			opos -= dz->buflen;
		}
	}
	*obufpos = opos;
	return FINISHED;
}
