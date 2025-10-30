<?php
/**
 * Alignment Settings Module
 * Settings > Widget Options :: Custom Alignment
 *
 * @copyright   Copyright (c) 2016, Jeffrey Carandang
 * @since       3.0
*/

// Exit if accessed directly
if ( ! defined( 'ABSPATH' ) ) exit;

/**
 * Create Card Module for Custom Alignment Options
 *
 * @since 3.0
 * @global $widget_options
 * @return void
 */

  /*
 * Note: Please add a class "no-settings" in the <li> card if the card has no additional configuration, if there are configuration please remove the class
 */

if( !function_exists( 'widgetopts_settings_alignment' ) ):
	function widgetopts_settings_alignment(){
	    global $widget_options; ?>
		<li class="widgetopts-module-card widgetopts-module-card-no-settings no-settings <?php echo ( $widget_options['alignment'] == 'activate' ) ? 'widgetopts-module-type-enabled' : 'widgetopts-module-type-disabled'; ?>" id="widgetopts-module-card-alignment" data-module-id="alignment">
			<div class="widgetopts-module-card-content">
				<h2><?php _e( 'Custom Alignment', 'widget-options' );?></h2>
				<p class="widgetopts-module-desc">
					<?php _e( 'Easily assign custom alignments on each widgets which will be relected on all devices.', 'widget-options' );?>
				</p>

				<div class="widgetopts-module-actions hide-if-no-js">
					<?php if( $widget_options['alignment'] == 'activate' ){ ?>
						<button class="button button-secondary widgetopts-toggle-settings"><?php _e( 'Learn More', 'widget-options' );?></button>
						<button class="button button-secondary widgetopts-toggle-activation"><?php _e( 'Disable', 'widget-options' );?></button>
					<?php }else{ ?>
						<button class="button button-secondary widgetopts-toggle-settings"><?php _e( 'Learn More', 'widget-options' );?></button>
						<button class="button button-primary widgetopts-toggle-activation"><?php _e( 'Enable', 'widget-options' );?></button>
					<?php } ?>

				</div>
			</div>

			<?php widgetopts_modal_start( $widget_options['alignment'] ); ?>
				<span class="dashicons widgetopts-dashicons widgetopts-no-top dashicons-editor-aligncenter"></span>
				<h3 class="widgetopts-modal-header"><?php _e( 'Custom Alignment', 'widget-options' );?></h3>
				<p>
					<?php _e( 'Custom alignment widget options will allow you to assign different content alignments for each widgets. You can choose whether you want them to be left, right, justify or centered aligned.', 'widget-options' );?>
				</p>
				<p class="widgetopts-settings-section">
					<?php _e( 'No additional settings available.', 'widget-options' );?>
				</p>
			<?php widgetopts_modal_end( $widget_options['alignment'] ); ?>

		</li>
	    <?php
	}
	add_action( 'widgetopts_module_cards', 'widgetopts_settings_alignment', 30 );
endif;
?>
